"""
This file contains the modules which perform the MCMC spectral imaging constraint. 
"""

#==================================================
# Requested imports
#==================================================

import sys
import pickle
import copy
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interpn
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates.sky_coordinate import SkyCoord
import emcee
import ctools
import gammalib
import warnings
import glob
from multiprocessing import Pool, cpu_count

from minot.model_tools import trapz_loglog
from minot.ClusterTools import map_tools
from kesacco.Tools import plotting
from kesacco.Tools import mcmc_common
from kesacco.Tools import make_cluster_template
from kesacco.Tools import cubemaking


#==================================================
# Build model grid
#==================================================

def build_model_grid(cpipe,
                     subdir,
                     rad, prof_ini,
                     spatial_value,
                     spatial_idx,
                     spectral_value,
                     spectral_idx,
                     bkg_spectral_value,
                     bkg_spectral_idx,
                     ps_spectral_value,
                     ps_spectral_idx,
                     includeIC=False,
                     rm_tmp=False,
                     start_num=0):
    """
    Build a grid of models for the cluster and background
        
    Parameters
    ----------
    - cpipe (kesacco object): a kesacco object
    - subdir (str): full path to the working subdirectory
    - rad (np array): the radius array for the 3d profile sampling
    - prof_ini (np array): the initial cluster profile to be rescaled
    - spatial_value (np array): the spatial rescaling values
    - spatial_idx (np array): the index corresponding to spatial_value
    - spectral_value (np array): the spectral rescaling values
    - spectral_idx (np array): the spectral corresponding to spectral_value
    - bkg_spectral_value (np array): the spectral rescaling values for the background
    - bkg_spectral_idx (np array): the spectral corresponding to spectral_value 
    for the background
    - ps_spectral_value (np array): the spectral rescaling values for the 
    point sources
    - ps_spectral_idx (np array): the spectral corresponding to spectral_value 
    for the point source
    - includeIC (bool): include inverse Compton in the model
    - rm_tmp (bool): remove temporary files
    - start_num (int): starting number for the template building

    Output
    ------
    - All temporary file (e.g. cluster templates, likelihood fit models, ...)
    and the Grid_Sampling.fits file

    """
    
    # Save the cluster model before modification
    cluster_tmp = copy.deepcopy(cpipe.cluster)

    # Number of points
    spatial_npt = len(spatial_value)
    spectral_npt = len(spectral_value)
    bkg_spectral_npt = len(bkg_spectral_value)
    ps_spectral_npt = len(ps_spectral_value)
    
    #===== Loop over all cluster models to be tested
    for imod in range(spatial_npt):
        for jmod in range(spectral_npt):
            special_condition = 1+jmod+imod*spectral_npt >= start_num
            #special_condition = True
            if special_condition:
                cl_tmp = str(1+jmod+imod*spectral_npt)+'/'+str(spatial_npt*spectral_npt)
                print('--- Building cluster template '+cl_tmp)
                print('    ---> spectral = '+str(spectral_value[jmod])+', spatial = '+str(spatial_value[imod]))

                #---------- Indexing
                spatial_i = spatial_value[imod]            
                spectral_j   = spectral_value[jmod]
                extij = 'TMP_'+str(imod)+'_'+str(jmod)
                    
                fexist = os.path.exists(subdir+'/Model_Cluster_Cube_'+extij+'.fits')
                if fexist:
                    print(subdir+'/Model_Cluster_Cube_'+extij+'.fits already exist, skip it')
                else:
                    #---------- Compute the model spectrum, map, and xml model file
                    # Re-scaling        
                    cluster_tmp.density_crp_model  = {'name':'User',
                                                      'radius':rad, 'profile':prof_ini.value ** spatial_i}
                    if cluster_tmp.spectrum_crp_model['name'] == 'PowerLaw':
                        cluster_tmp.spectrum_crp_model = {'name':'PowerLaw', 'Index':spectral_j}
                    elif cluster_tmp.spectrum_crp_model['name'] == 'MomentumPowerLaw':
                        cluster_tmp.spectrum_crp_model = {'name':'MomentumPowerLaw',
                                                          'Index':spectral_j,
                                                          'Mass':cluster_tmp.spectrum_crp_model['Mass']}
                    else:
                        raise ValueError('Only PowerLaw and MomentumPowerLaw implemented here')
                
                    # Cluster model
                    make_cluster_template.make_map(cluster_tmp,
                                                   subdir+'/Model_Cluster_Map_'+extij+'.fits',
                                                   Egmin=cpipe.obs_setup.get_emin(),
                                                   Egmax=cpipe.obs_setup.get_emax(),
                                                   includeIC=includeIC)
                    make_cluster_template.make_spectrum(cluster_tmp,
                                                        subdir+'/Model_Cluster_Spectrum_'+extij+'.txt',
                                                        energy=np.logspace(-1,5,1000)*u.GeV,
                                                        includeIC=includeIC)
           
                    # xml model
                    model_tot = gammalib.GModels(cpipe.output_dir+'/Ana_Model_Input_Stack.xml')
                    clencounter = 0
                    for i in range(len(model_tot)):
                        if model_tot[i].name() == cluster_tmp.name:
                            spefn = subdir+'/Model_Cluster_Spectrum_'+extij+'.txt'
                            model_tot[i].spectral().filename(spefn)
                            spafn = subdir+'/Model_Cluster_Map_'+extij+'.fits'
                            model_tot[i].spatial(gammalib.GModelSpatialDiffuseMap(spafn))
                            clencounter += 1
                    if clencounter != 1:
                        raise ValueError('No cluster encountered in the input stack model')
                    model_tot.save(subdir+'/Model_Cluster_'+extij+'.xml')
           
                    # Remove background and point sources
                    cpipe._rm_source_xml(subdir+'/Model_Cluster_'+extij+'.xml',
                                         subdir+'/Model_Cluster_'+extij+'.xml',
                                         'BackgroundModel')
                    
                    for i in range(len(cpipe.compact_source.name)):
                        cpipe._rm_source_xml(subdir+'/Model_Cluster_'+extij+'.xml',
                                             subdir+'/Model_Cluster_'+extij+'.xml',
                                             cpipe.compact_source.name[i])
                    
                    #---------- Compute the 3D cluster cube            
                    modcube = cubemaking.model_cube(cpipe.output_dir,
                                                    cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                                    cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                                    cpipe.spec_ebinalg,
                                                    edisp=cpipe.spec_edisp,
                                                    stack=cpipe.method_stack,
                                                    silent=True,
                                                    logfile=subdir+'/Model_Cluster_Cube_log_'+extij+'.txt',
                                                    inmodel_usr=subdir+'/Model_Cluster_'+extij+'.xml',
                                                    outmap_usr=subdir+'/Model_Cluster_Cube_'+extij+'.fits')
    
    #===== Loop over all background models to be tested
    for jmod in range(bkg_spectral_npt):
        print('--- Building background template '+str(1+jmod)+'/'+str(bkg_spectral_npt))
        print('    ---> spectral value = '+str(bkg_spectral_value[jmod]))
            
        #---------- Indexing
        spectral_j   = bkg_spectral_value[jmod]
        extij = 'TMP_'+str(jmod)

        fexist = os.path.exists(subdir+'/Model_Background_Cube_'+extij+'.fits')
        if fexist:
            print(subdir+'/Model_Background_Cube_'+extij+'.fits already exist, skip it')
        else:
            #---------- xml model
            model_tot = gammalib.GModels(cpipe.output_dir+'/Ana_Model_Input_Stack.xml')
            bkencounter = 0
            for i in range(len(model_tot)):
                if model_tot[i].name() == 'BackgroundModel':
                    model_tot[i].spectral().index(model_tot[i].spectral().index() + spectral_j)
                    bkencounter += 1
                    
            if bkencounter != 1:
                raise ValueError('No background encountered in the input stack model')
            model_tot.save(subdir+'/Model_Background_'+extij+'.xml')
            
            # Remove Cluster and point sources
            cpipe._rm_source_xml(subdir+'/Model_Background_'+extij+'.xml',
                                 subdir+'/Model_Background_'+extij+'.xml',
                                 cpipe.cluster.name)
            
            for i in range(len(cpipe.compact_source.name)):
                cpipe._rm_source_xml(subdir+'/Model_Background_'+extij+'.xml',
                                     subdir+'/Model_Background_'+extij+'.xml',
                                     cpipe.compact_source.name[i])
         
            #---------- Compute the 3D background cube            
            modcube = cubemaking.model_cube(cpipe.output_dir,
                                            cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                            cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                            cpipe.spec_ebinalg,
                                            edisp=cpipe.spec_edisp,
                                            stack=cpipe.method_stack,
                                            silent=True,
                                            logfile=subdir+'/Model_Background_Cube_log_'+extij+'.txt',
                                            inmodel_usr=subdir+'/Model_Background_'+extij+'.xml',
                                            outmap_usr=subdir+'/Model_Background_Cube_'+extij+'.fits')
    
    #===== Loop over all point source models to be tested
    for ips in range(len(cpipe.compact_source.name)):
        print('------ Point source '+cpipe.compact_source.name[ips])
        for jmod in range(ps_spectral_npt):
            print('--- Building point source template '+str(1+jmod)+'/'+str(ps_spectral_npt))
            print('    ---> spectral value = '+str(ps_spectral_value[jmod]))

            #---------- Indexing
            spectral_j   = ps_spectral_value[jmod]
            extij = 'TMP_'+str(jmod)


            fexist = os.path.exists(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_Cube_'+extij+'.fits')
            if fexist:
                print(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_Cube_'+extij+'.fits already exist, skip')
            else:
                #---------- xml model
                model_tot = gammalib.GModels(cpipe.output_dir+'/Ana_Model_Input_Stack.xml')
                psencounter = 0
                for i in range(len(model_tot)):
                    if model_tot[i].name() == cpipe.compact_source.name[ips]:
                        #model_tot[i].spectral().index(model_tot[i].spectral().index() + spectral_j)
                        model_tot[i].spectral()[1].value(model_tot[i].spectral()[1].value() + spectral_j)
                        psencounter += 1
    
                if psencounter != 1:
                    raise ValueError('No point source encountered in the input stack model')
                model_tot.save(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml')
            
                # Remove Cluster and background
                cpipe._rm_source_xml(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                     subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                     cpipe.cluster.name)
                
                cpipe._rm_source_xml(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                     subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                     'BackgroundModel')
    
                # Remove other point sources
                for jps in range(len(cpipe.compact_source.name)):
                    if jps != ips:
                        cpipe._rm_source_xml(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                             subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                             cpipe.compact_source.name[jps])
                
                #---------- Compute the 3D background cube            
                modcube = cubemaking.model_cube(cpipe.output_dir,
                                                cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                                cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                                cpipe.spec_ebinalg,
                                                edisp=cpipe.spec_edisp,
                                                stack=cpipe.method_stack,
                                                silent=True,
                                                logfile=subdir+'/Model_'+cpipe.compact_source.name[ips]+'_Cube_log_'+extij+'.txt',
                                                inmodel_usr=subdir+'/Model_'+cpipe.compact_source.name[ips]+'_'+extij+'.xml',
                                                outmap_usr=subdir+'/Model_'+cpipe.compact_source.name[ips]+'_Cube_'+extij+'.fits')
    
    #===== Build the grid
    # Get the grid info
    hdul = fits.open(subdir+'/Model_Cluster_Cube_TMP_'+str(0)+'_'+str(0)+'.fits')
    head = hdul[0].header
    hdul.close()
    
    Npixx = head['NAXIS1']
    Npixy = head['NAXIS2']
    NpixE = head['NAXIS3']

    # Build the cluster grid
    modgrid_cl = np.zeros((spatial_npt, spectral_npt, NpixE, Npixy, Npixx))
    for imod in range(spatial_npt):
        for jmod in range(spectral_npt):
            extij = 'TMP_'+str(imod)+'_'+str(jmod)
            hdul = fits.open(subdir+'/Model_Cluster_Cube_'+extij+'.fits')
            modgrid_cl[imod,jmod,:,:,:] = hdul[0].data
            hdul.close()
    grid_cl_hdu = fits.ImageHDU(modgrid_cl)
            
    # Build the background grid
    modgrid_bk = np.zeros((bkg_spectral_npt, NpixE, Npixy, Npixx))
    for jmod in range(bkg_spectral_npt):
        extij = 'TMP_'+str(jmod)
        hdul = fits.open(subdir+'/Model_Background_Cube_'+extij+'.fits')
        modgrid_bk[jmod,:,:,:] = hdul[0].data
        hdul.close()
    grid_bk_hdu = fits.ImageHDU(modgrid_bk)

    # Build the background grid
    gridlist_ps_hdu = []
    for ips in range(len(cpipe.compact_source.name)):
        modgrid_ps = np.zeros((ps_spectral_npt, NpixE, Npixy, Npixx))
        for jmod in range(ps_spectral_npt):
            extij = 'TMP_'+str(jmod)
            hdul = fits.open(subdir+'/Model_'+cpipe.compact_source.name[ips]+'_Cube_'+extij+'.fits')
            modgrid_ps[jmod,:,:,:] = hdul[0].data
            hdul.close()
            grid_ps_hdu = fits.ImageHDU(modgrid_ps)
        gridlist_ps_hdu.append(grid_ps_hdu)

    #===== Save in a table
    scal_spa = Table()
    scal_spa['spatial_idx'] = spatial_idx
    scal_spa['spatial_val'] = spatial_value
    scal_spa_hdu = fits.BinTableHDU(scal_spa)
    scal_spe = Table()
    scal_spe['spectral_idx'] = spectral_idx
    scal_spe['spectral_val'] = spectral_value
    scal_spe_hdu = fits.BinTableHDU(scal_spe)

    bk_scal_spe = Table()
    bk_scal_spe['bk_spectral_idx'] = bkg_spectral_idx
    bk_scal_spe['bk_spectral_val'] = bkg_spectral_value
    bk_scal_spe_hdu = fits.BinTableHDU(bk_scal_spe)

    ps_scal_spe = Table()
    ps_scal_spe['ps_spectral_idx'] = ps_spectral_idx
    ps_scal_spe['ps_spectral_val'] = ps_spectral_value
    ps_scal_spe_hdu = fits.BinTableHDU(ps_scal_spe)
    
    hdul = fits.HDUList()
    hdul.append(scal_spa_hdu)
    hdul.append(scal_spe_hdu)
    hdul.append(grid_cl_hdu)
    hdul.append(bk_scal_spe_hdu)
    hdul.append(grid_bk_hdu)
    hdul.append(ps_scal_spe_hdu)
    for ips in range(len(cpipe.compact_source.name)):
        hdul.append(gridlist_ps_hdu[ips])
    hdul.writeto(subdir+'/Grid_Sampling.fits', overwrite=True)

    #===== Save the properties of the last computation run
    np.save(subdir+'/Grid_Parameters.npy',
            [cpipe.cluster, spatial_value, spectral_value], allow_pickle=True)
    
    #===== remove TMP files
    if rm_tmp:
        tmp_list = glob.glob(subdir+"/Model_*")
        for fi in tmp_list:
            os.remove(fi)

#==================================================
# Compute uncertainty on the model grid
#==================================================

def validation_model_grid_itpl(param_test, input_files, cpipe, subdir, rad, prof_ini, includeIC=False):
    """
    Validate the model interpolation for a set of parameters. To to so
    it uses the interpolation on one hand (quick), and also compute the 
    exact model without interpolation as done in build_model_grid.
        
    Parameters
    ----------
    - param_test (list):
    - input_files (str list):
    - cpipe (kesacco object):
    - subdir (str):
    - rad (np array):
    - prof_ini (np array):
    - includeIC (bool):

    Output
    ------
    - Plot the relative difference between truth and interpolated model

    """

    if param_test[5] != param_test[7] or param_test[6] != param_test[8]:
        raise ValueError('The point source parameter should be the same for all point source in this test')
    ext_name = str(param_test[1])+'_'+str(param_test[2])+'_'+str(param_test[4])+'_'+str(param_test[6])
    
    #---------- Build a subsubdirectory for the test
    if not os.path.exists(subdir+'/ValidationItpl'):
        os.mkdir(subdir+'/ValidationItpl')

    #---------- Build the MCMC model from interpolation
    data, modgrid = read_data(input_files)
    model_mcmc = model_specimg(modgrid, param_test)

    #---------- Build the true model
    build_model_grid(cpipe, subdir+'/ValidationItpl', rad, prof_ini,
                     np.array([param_test[1]]), np.array([0]),
                     np.array([param_test[2]]), np.array([0]),
                     np.array([param_test[4]]), np.array([0]),
                     np.array([param_test[6]]), np.array([0]),
                     includeIC=includeIC,
                     rm_tmp=False)
    
    #---------- How many point sources
    Nps   = len(cpipe.compact_source.name)
    Nebin = model_mcmc['cluster'].shape[0]

    #----------- Show the difference (cluster)
    hdu = fits.open(subdir+'/ValidationItpl/Model_Cluster_Cube_TMP_0_0.fits')
    model_cl_true  = hdu[0].data
    hdu.close()
    
    pdf_pages = PdfPages(subdir+'/ValidationItpl/Cluster_par_'+ext_name+'.pdf')
    for ibin in range(Nebin):
        fig = plt.figure(0, figsize=(15, 10))
        ax = plt.subplot(2,3,1)
        vmin = np.nanmin(model_cl_true[ibin,:,:])
        vmax = np.nanmax(model_cl_true[ibin,:,:])
        plt.imshow(model_cl_true[ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('True model')
        plt.colorbar()
        
        ax = plt.subplot(2,3,2)
        plt.imshow(model_mcmc['cluster'][ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Interpolated model')
        plt.colorbar()
        
        ax = plt.subplot(2,3,3)
        plt.imshow((model_mcmc['cluster'][ibin,:,:]-model_cl_true[ibin,:,:]),origin='lower', cmap='rainbow')
        plt.title('Difference (itpl-true)')
        plt.colorbar()
        
        ax = plt.subplot(2,3,4)
        syst = (model_mcmc['cluster'][ibin,:,:]-model_cl_true[ibin,:,:])/model_cl_true[ibin,:,:]*100
        plt.imshow(syst, origin='lower', cmap='rainbow')
        plt.title('Difference/true x 100')
        plt.colorbar()
        
        ax = plt.subplot(2,3,5)
        syst = (model_mcmc['cluster'][ibin,:,:]-model_cl_true[ibin,:,:])/model_cl_true[ibin,:,:]*100
        vmin = np.mean(syst[np.isfinite(syst)]) - 2*np.std(syst[np.isfinite(syst)])
        vmax = np.mean(syst[np.isfinite(syst)]) + 2*np.std(syst[np.isfinite(syst)])
        plt.imshow(syst, origin='lower', vmin=vmin, vmax=vmax, cmap='rainbow')
        plt.title(r'Difference/true x 100 ($\pm 2 \sigma$ clipped)')
        plt.colorbar()
        
        ax = plt.subplot(2,3,6)
        syst = (model_mcmc['cluster'][ibin,:,:]-model_cl_true[ibin,:,:])/model_cl_true[ibin,:,:]**0.5
        plt.imshow(syst, origin='lower', cmap='rainbow')
        plt.title('Difference/true$^{1/2}$')
        plt.colorbar()
        
        pdf_pages.savefig(fig)
        plt.close()
    pdf_pages.close()

    #----------- Show the difference (background)
    hdu = fits.open(subdir+'/ValidationItpl/Model_Background_Cube_TMP_0.fits')
    model_bk_true  = hdu[0].data
    hdu.close()

    pdf_pages = PdfPages(subdir+'/ValidationItpl/Background_par_'+ext_name+'.pdf')
    for ibin in range(Nebin):
        fig = plt.figure(0, figsize=(15, 10))
        ax = plt.subplot(2,3,1)
        vmin = np.nanmin(model_bk_true[ibin,:,:])
        vmax = np.nanmax(model_bk_true[ibin,:,:])
        plt.imshow(model_bk_true[ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('True model')
        plt.colorbar()
        
        ax = plt.subplot(2,3,2)
        plt.imshow(model_mcmc['background'][ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Interpolated model')
        plt.colorbar()

        ax = plt.subplot(2,3,3)
        plt.imshow((model_mcmc['background'][ibin,:,:]-model_bk_true[ibin,:,:]), origin='lower', cmap='rainbow')
        plt.title('Difference (itpl-true)')
        plt.colorbar()
                
        ax = plt.subplot(2,3,4)
        syst = (model_mcmc['background'][ibin,:,:]-model_bk_true[ibin,:,:])/model_bk_true[ibin,:,:]*100
        plt.imshow(syst, origin='lower', cmap='rainbow')
        plt.title('Difference/true x 100')
        plt.colorbar()

        ax = plt.subplot(2,3,5)
        syst = (model_mcmc['background'][ibin,:,:]-model_bk_true[ibin,:,:])/model_bk_true[ibin,:,:]*100
        vmin = np.mean(syst[np.isfinite(syst)]) - 2*np.std(syst[np.isfinite(syst)])
        vmax = np.mean(syst[np.isfinite(syst)]) + 2*np.std(syst[np.isfinite(syst)])
        plt.imshow(syst, vmin=vmin, vmax=vmax, origin='lower', cmap='rainbow')
        plt.title(r'Difference/true x 100 ($\pm 2 \sigma$ clipped)')
        plt.colorbar()

        ax = plt.subplot(2,3,6)
        syst = (model_mcmc['background'][ibin,:,:]-model_bk_true[ibin,:,:])/model_bk_true[ibin,:,:]**0.5
        plt.imshow(syst, origin='lower', cmap='rainbow')
        plt.title('Difference/true$^{1/2}$')
        plt.colorbar()
        
        pdf_pages.savefig(fig)
        plt.close()
    pdf_pages.close()
    
    #----------- Show the difference (point sources)
    for ips in range(Nps):
        srcname = cpipe.compact_source.name[ips]

        hdu = fits.open(subdir+'/ValidationItpl/Model_'+srcname+'_Cube_TMP_0.fits')
        model_ps_true  = hdu[0].data
        hdu.close()        

        pdf_pages = PdfPages(subdir+'/ValidationItpl/'+srcname+'_par_'+ext_name+'.pdf')
        for ibin in range(Nebin):
            fig = plt.figure(0, figsize=(15, 10))
            ax = plt.subplot(2,3,1)
            vmin = np.nanmin(model_mcmc['point_sources'][ips][ibin,:,:])
            vmax = np.nanmax(model_mcmc['point_sources'][ips][ibin,:,:])
            plt.imshow(model_mcmc['point_sources'][ips][ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title('Interpolated model')
            plt.colorbar()
            
            ax = plt.subplot(2,3,2)
            plt.imshow(model_ps_true[ibin,:,:], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title('True model')
            plt.colorbar()
            
            ax = plt.subplot(2,3,3)
            plt.imshow((model_mcmc['point_sources'][ips][ibin,:,:]-model_ps_true[ibin,:,:]), origin='lower', cmap='rainbow')
            plt.title('Difference (itpl-true)')
            plt.colorbar()

            ax = plt.subplot(2,3,4)
            syst = (model_mcmc['point_sources'][ips][ibin,:,:]-model_ps_true[ibin,:,:])/model_ps_true[ibin,:,:]*100
            plt.imshow(syst, origin='lower', cmap='rainbow')
            plt.title('Difference/true x 100')
            plt.colorbar()
            
            ax = plt.subplot(2,3,5)
            syst = (model_mcmc['point_sources'][ips][ibin,:,:]-model_ps_true[ibin,:,:])/model_ps_true[ibin,:,:]*100
            vmin = np.mean(syst[np.isfinite(syst)]) - 2*np.std(syst[np.isfinite(syst)])
            vmax = np.mean(syst[np.isfinite(syst)]) + 2*np.std(syst[np.isfinite(syst)])
            plt.imshow(syst, vmin=vmin, vmax=vmax, origin='lower', cmap='rainbow')
            plt.title(r'Difference/true x 100 ($\pm 2 \sigma$ clipped)')
            plt.colorbar()
            
            ax = plt.subplot(2,3,6)
            syst = (model_mcmc['point_sources'][ips][ibin,:,:]-model_ps_true[ibin,:,:])/model_ps_true[ibin,:,:]**0.5
            plt.imshow(syst, origin='lower', cmap='rainbow')
            plt.title('Difference/true$^{1/2}$')
            plt.colorbar()
            
            pdf_pages.savefig(fig)
            plt.close()
        pdf_pages.close()
        
            
#==================================================
# Get models from the parameter space
#==================================================

def get_mc_model(modgrid, param_chains, Nmc=100):
    """
    Get models randomly sampled from the parameter space
        
    Parameters
    ----------
    - modgrid (array): grid of model
    - param_chains (ndarray): array of chains parametes
    - Nmc (int): number of models

    Output
    ------
    - MC_model (ndarray): Nmc x N_eng array

    """

    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1],
                                    param_chains.shape[2])
    
    Nsample = len(par_flat[:,0])-1
    
    MC_model_cluster = np.zeros((Nmc, modgrid['xx_val'].shape[0],
                                 modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2]))
    MC_model_background = np.zeros((Nmc, modgrid['xx_val'].shape[0],
                                    modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2]))
    MC_model_ps = []
    for ips in range(len(modgrid['models_ps_list'])):
        MC_model_ps.append(np.zeros((Nmc, modgrid['xx_val'].shape[0],
                                     modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2])))
    MC_model_total = np.zeros((Nmc, modgrid['xx_val'].shape[0],
                               modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2]))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        mods = model_specimg(modgrid, param_MC)
        MC_model_cluster[i,:,:,:]    = mods['cluster']
        MC_model_background[i,:,:,:] = mods['background']
        for ips in range(len(modgrid['models_ps_list'])):
            MC_model_ps[ips][i,:,:,:] = mods['point_sources'][ips]
        MC_model_total[i,:,:,:] = mods['total']

    MC_models = {'cluster':MC_model_cluster,
                 'background':MC_model_background,
                 'point_sources':MC_model_ps,
                 'total':MC_model_total}
    
    return MC_models


#==================================================
# Plot the output fit model
#==================================================

def modelplot(data, modbest, MC_model, corner_mask, header, Ebins, outdir,
              conf=68.0, FWHM=0.1*u.deg,
              theta=1*u.deg, coord=None, profile_reso=0.05*u.deg):
    """
    Plot the data versus model and constraints
        
    Parameters
    ----------
    - data (dict): data file
    - modbest (dict): best fit model
    - MC_model (dict): Monte Carlo models computed with get_mc_model
    - header (str): map header
    - Ebins (ndarray): Energy bins 
    - outdir (str): path to output directory
    - conf (float): confidence limit used in plots
    - FWHM (quantity): smoothing FWHM
    - theta (quantity): angle used for spectral extraction
    - coord (SkyCoord): coordinates of the cluster
    - profile_reso (quantity): bin used for profile extraction
    
    Output
    ------
    Plots are saved
    
    """
    
    #========== Get needed information
    reso = header['CDELT2']
    sigma_sm = (FWHM/(2*np.sqrt(2*np.log(2)))).to_value('deg')/reso
    header['NAXIS'] = 2
    del header['NAXIS3']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proj = WCS(header)
        ra_map, dec_map = map_tools.get_radec_map(header)

    if coord is None:
        coord = SkyCoord(np.median(ra_map)*u.deg, np.median(dec_map)*u.deg, frame='icrs')

    radmap = map_tools.greatcircle(ra_map, dec_map,
                                   coord.ra.to_value('deg'), coord.dec.to_value('deg'))
    
    #========== Plot 1: map, Data - model, stack
    fig = plt.figure(0, figsize=(10, 7))
    ax = plt.subplot(221, projection=proj)
    vmin = np.nanmin(gaussian_filter(np.sum(data, axis=0), sigma=sigma_sm))
    vmax = np.nanmax(gaussian_filter(np.sum(data, axis=0), sigma=sigma_sm))
    plt.imshow(gaussian_filter(np.sum(data, axis=0), sigma=sigma_sm)*corner_mask[0,:,:],
                origin='lower', cmap='magma',norm=SymLogNorm(1, base=10, vmin=vmin, vmax=vmax))
    cb = plt.colorbar()
    plt.title('Data (counts)')
    plt.xlabel(' ')
    plt.ylabel('Dec.')
    
    ax = plt.subplot(222, projection=proj)
    plt.imshow(gaussian_filter(np.sum(modbest['total'],axis=0), sigma=sigma_sm)*corner_mask[0,:,:],
               origin='lower', cmap='magma', norm=SymLogNorm(1, base=10, vmin=cb.norm.vmin, vmax=cb.norm.vmax))
    plt.colorbar()
    plt.title('Model (counts)')
    plt.xlabel(' ')
    plt.ylabel(' ')

    cntmap_ps   = data*0
    for ips in range(len(modbest['point_sources'])):
        cntmap_ps += modbest['point_sources'][ips]
    vmin0 = np.amin(gaussian_filter(np.sum(data-modbest['background']-cntmap_ps, axis=0),sigma=sigma_sm))
    vmax0 = np.amax(gaussian_filter(np.sum(data-modbest['background']-cntmap_ps, axis=0),sigma=sigma_sm))
    vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])
    ax = plt.subplot(223, projection=proj)
    plt.imshow(gaussian_filter(np.sum(data-modbest['background']-cntmap_ps, axis=0), sigma=sigma_sm)*corner_mask[0,:,:],
               origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
    plt.colorbar()
    filt1 = gaussian_filter(np.sum(data-modbest['background']-cntmap_ps, axis=0), sigma=sigma_sm)
    filt2 = np.sqrt(gaussian_filter(np.sum(modbest['total'], axis=0), sigma=sigma_sm))
    snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2 * corner_mask[0,:,:]
    cont = ax.contour(snr, levels=np.array([-6,-4,-2,2,4,6,8,10]), colors='black')
    plt.title('Data - background (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')

    vmin0 = np.amin(gaussian_filter(np.sum(data-modbest['total'], axis=0),sigma=sigma_sm)*corner_mask[0,:,:])
    vmax0 = np.amax(gaussian_filter(np.sum(data-modbest['total'], axis=0),sigma=sigma_sm)*corner_mask[0,:,:])
    vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])
    ax = plt.subplot(224, projection=proj)
    plt.imshow(gaussian_filter(np.sum(data-modbest['total'], axis=0), sigma=sigma_sm)*corner_mask[0,:,:],
               origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
    plt.colorbar()
    filt1 = gaussian_filter(np.sum(data-modbest['total'], axis=0), sigma=sigma_sm)
    filt2 = np.sqrt(gaussian_filter(np.sum(modbest['total'], axis=0), sigma=sigma_sm))
    snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2 *corner_mask[0,:,:]
    cont = ax.contour(snr, levels=np.array([-6,-4,-2,2,4,6,8,10]), colors='black')
    plt.title('Residual (counts)')
    plt.xlabel('R.A.')
    plt.ylabel(' ')

    plt.savefig(outdir+'/MCMC_MapResidual.pdf')
    plt.close()
    
    #========== Plot 2: map, Data - model, energy bins
    pdf_pages = PdfPages(outdir+'/MCMC_MapSliceResidual.pdf')
    
    for i in range(len(Ebins)):
        Ebinprint = '{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)
    
        fig = plt.figure(0, figsize=(10, 7))
        ax = plt.subplot(221, projection=proj)
        vmin = np.nanmin(gaussian_filter(data[i,:,:], sigma=sigma_sm))
        vmax = np.nanmax(gaussian_filter(data[i,:,:], sigma=sigma_sm))
        plt.imshow(gaussian_filter(data[i,:,:], sigma=sigma_sm)*corner_mask[0,:,:],
                   origin='lower', cmap='magma', norm=SymLogNorm(1, base=10, vmin=vmin, vmax=vmax))
        cb = plt.colorbar()
        plt.title('Data (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel(' ')
        plt.ylabel('Dec.')
        
        ax = plt.subplot(222, projection=proj)
        plt.imshow(gaussian_filter(modbest['total'][i,:,:], sigma=sigma_sm)*corner_mask[0,:,:],
                   origin='lower', cmap='magma', norm=SymLogNorm(1, base=10,vmin=cb.norm.vmin, vmax=cb.norm.vmax))
        plt.colorbar()
        plt.title('Model (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel(' ')
        plt.ylabel(' ')

        cntmap_ps   = data[i,:,:]*0
        for ips in range(len(modbest['point_sources'])):
            cntmap_ps += (modbest['point_sources'][ips])[i,:,:]
        
        vmin0 = np.amin(gaussian_filter((data-modbest['background'])[i,:,:]-cntmap_ps, sigma=sigma_sm))
        vmax0 = np.amax(gaussian_filter((data-modbest['background'])[i,:,:]-cntmap_ps, sigma=sigma_sm))
        vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])
        
        ax = plt.subplot(223, projection=proj)
        plt.imshow(gaussian_filter((data-modbest['background'])[i,:,:]-cntmap_ps, sigma=sigma_sm)*corner_mask[0,:,:],
                   origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
        filt1 = gaussian_filter((data-modbest['background'])[i,:,:]-cntmap_ps, sigma=sigma_sm)
        filt2 = np.sqrt(gaussian_filter(modbest['total'][i,:,:], sigma=sigma_sm))
        snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2 *corner_mask[0,:,:]
        cont = ax.contour(snr, levels=np.array([-9,-7,-5,-3,3,5,7,9]), colors='black')
        plt.colorbar()
        plt.title('Data - background (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        
        vmin0 = np.amin(gaussian_filter((data-modbest['total'])[i,:,:], sigma=sigma_sm)*corner_mask[0,:,:])
        vmax0 = np.amax(gaussian_filter((data-modbest['total'])[i,:,:], sigma=sigma_sm)*corner_mask[0,:,:])
        vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])
        
        ax = plt.subplot(224, projection=proj)
        plt.imshow(gaussian_filter((data-modbest['total'])[i,:,:], sigma=sigma_sm)*corner_mask[0,:,:],
                   origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
        filt1 = gaussian_filter((data-modbest['total'])[i,:,:], sigma=sigma_sm)
        filt2 = np.sqrt(gaussian_filter(modbest['total'][i,:,:], sigma=sigma_sm))
        snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2 * corner_mask[0,:,:]
        cont = ax.contour(snr, levels=np.array([-9,-7,-5,-3,3,5,7,9]), colors='black')
        plt.colorbar()
        plt.title('Residual (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel(' ')

        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()
    
    #========== Plot 3: profile, background subtracted, stack
    cntmap_data = np.sum(data*corner_mask, axis=0)
    cntmap_tot  = np.sum(modbest['total']*corner_mask, axis=0)
    cntmap_cl   = np.sum(modbest['cluster']*corner_mask, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_dat, p_dat, err_dat = map_tools.radial_profile_cts(cntmap_data,
                                                             [coord.icrs.ra.to_value('deg'),
                                                              coord.icrs.dec.to_value('deg')],
                                                             stddev=np.sqrt(cntmap_tot),
                                                             header=header,
                                                             binsize=profile_reso.to_value('deg'),
                                                             stat='POISSON', counts2brightness=True,
                                                             residual=True)
        r_tot, p_tot, err_tot = map_tools.radial_profile_cts(cntmap_tot,
                                                             [coord.icrs.ra.to_value('deg'),
                                                              coord.icrs.dec.to_value('deg')],
                                                             stddev=np.sqrt(cntmap_tot),
                                                             header=header,
                                                             binsize=profile_reso.to_value('deg'),
                                                             stat='POISSON', counts2brightness=True,
                                                             residual=True)
        r_cl, p_cl, err_cl = map_tools.radial_profile_cts(cntmap_cl,
                                                          [coord.icrs.ra.to_value('deg'),
                                                           coord.icrs.dec.to_value('deg')],
                                                          stddev=np.sqrt(cntmap_cl),
                                                          header=header,
                                                          binsize=profile_reso.to_value('deg'),
                                                          stat='POISSON', counts2brightness=True,
                                                          residual=True)
        Nmc = MC_model['cluster'].shape[0]
        p_cl_mc = np.zeros((Nmc, len(p_cl)))
        for i in range(Nmc):
            r_cl_i, p_cl_i, err_cl_i = map_tools.radial_profile_cts(np.sum(MC_model['cluster'][i,:,:,:]*corner_mask,axis=0),
                                                                    [coord.icrs.ra.to_value('deg'),
                                                                     coord.icrs.dec.to_value('deg')],
                                                                    stddev=np.sqrt(cntmap_cl),
                                                                    header=header,
                                                                    binsize=profile_reso.to_value('deg'),
                                                                    stat='POISSON', counts2brightness=True,
                                                                    residual=True)
            p_cl_mc[i, :]    = p_cl_i
        p_cl_up    = np.percentile(p_cl_mc, (100-conf)/2.0, axis=0)
        p_cl_lo    = np.percentile(p_cl_mc, 100 - (100-conf)/2.0, axis=0)

    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(r_dat)/1.1, np.amax(r_dat)*1.1]
    rngyp = 1.2*np.nanmax(p_dat-(p_tot-p_cl)+3*err_dat)
    rngym = 0.5*np.nanmin((p_dat-(p_tot-p_cl))[p_dat-(p_tot-p_cl) > 0])
    ylim = [rngym, rngyp]

    ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
    ax1.errorbar(r_dat, p_dat-(p_tot-p_cl), yerr=err_tot,
                 marker='o', elinewidth=2, color='k',
                 markeredgecolor="black", markerfacecolor="k",
                 ls ='', label='Data')
    ax1.plot(r_dat, p_cl_up, color='blue', linewidth=1, linestyle='--', label=str(conf)+'% C.L.')
    ax1.plot(r_dat, p_cl_lo, color='blue', linewidth=1, linestyle='--')
    ax1.fill_between(r_dat, p_cl_up, p_cl_lo, alpha=0.2, color='blue')
    for i in range(Nmc):
        ax1.plot(r_dat, p_cl_mc[i, :], color='blue', linewidth=1, alpha=0.05)    
    ax1.set_ylabel('Profile (deg$^{-2}$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xlim[0], xlim[1])
    #ax1.set_ylim(ylim[0], ylim[1])     <<<<<-------------->>>>>
    ax1.set_ylim(1e1, 1e5)
    ax1.set_xticks([])
    ax1.legend()
    
    # Add extra unit axes
    ax2 = ax1.twinx()
    ax2.plot(r_dat, (p_cl*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
    ax2.set_ylabel('Profile (arcmin$^{-2}$)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                 (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
    
    # Residual plot
    ax3.plot(r_dat, (p_dat-p_tot)/err_tot,linestyle='', marker='o', color='k')
    ax3.plot(r_dat,  r_dat*0, linestyle='-', color='k')
    ax3.plot(r_dat,  r_dat*0+2, linestyle='--', color='k')
    ax3.plot(r_dat,  r_dat*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('Radius (deg)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(outdir+'/MCMC_Profile_bkgsub.pdf')
    plt.close()

    #========== Plot 4: profile, background included, stack
    cntmap_data = np.sum(data*corner_mask, axis=0)
    cntmap_tot  = np.sum(modbest['total']*corner_mask, axis=0)
    cntmap_cl   = np.sum(modbest['cluster']*corner_mask, axis=0)
    cntmap_bk   = np.sum(modbest['background']*corner_mask, axis=0)
    cntmap_ps   = []
    for ips in range(len(modbest['point_sources'])):
        cntmap_ps.append(np.sum(modbest['point_sources'][ips], axis=0))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_dat, p_dat, err_dat = map_tools.radial_profile_cts(cntmap_data,
                                                             [coord.icrs.ra.to_value('deg'),
                                                              coord.icrs.dec.to_value('deg')],
                                                             stddev=np.sqrt(cntmap_tot),
                                                             header=header,
                                                             binsize=profile_reso.to_value('deg'),
                                                             stat='POISSON', counts2brightness=True,
                                                             residual=True)
        r_tot, p_tot, err_tot = map_tools.radial_profile_cts(cntmap_tot,
                                                             [coord.icrs.ra.to_value('deg'),
                                                              coord.icrs.dec.to_value('deg')],
                                                             stddev=np.sqrt(cntmap_tot),
                                                             header=header,
                                                             binsize=profile_reso.to_value('deg'),
                                                             stat='POISSON', counts2brightness=True,
                                                             residual=True)
        r_cl, p_cl, err_cl = map_tools.radial_profile_cts(cntmap_cl,
                                                          [coord.icrs.ra.to_value('deg'),
                                                           coord.icrs.dec.to_value('deg')],
                                                          stddev=np.sqrt(cntmap_tot),
                                                          header=header,
                                                          binsize=profile_reso.to_value('deg'),
                                                          stat='POISSON', counts2brightness=True,
                                                          residual=True)
        r_bk, p_bk, err_bk = map_tools.radial_profile_cts(cntmap_bk,
                                                          [coord.icrs.ra.to_value('deg'),
                                                           coord.icrs.dec.to_value('deg')],
                                                          stddev=np.sqrt(cntmap_tot),
                                                          header=header,
                                                          binsize=profile_reso.to_value('deg'),
                                                          stat='POISSON', counts2brightness=True,
                                                          residual=True)
        p_ps = []
        for ips in range(len(modbest['point_sources'])):
            r_psi, p_psi, err_psi = map_tools.radial_profile_cts(cntmap_ps[ips],
                                                                 [coord.icrs.ra.to_value('deg'),
                                                                  coord.icrs.dec.to_value('deg')],
                                                                 stddev=np.sqrt(cntmap_tot),
                                                                 header=header,
                                                                 binsize=profile_reso.to_value('deg'),
                                                                 stat='POISSON', counts2brightness=True,
                                                                 residual=True)
            p_ps.append(p_psi)
                  
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(r_dat)/1.1, np.amax(r_dat)*1.1]
    rngyp = 1.2*np.nanmax(p_dat+3*err_tot)
    rngym = 0.5*np.nanmin((p_dat)[p_dat > 0])
    ylim = [rngym, rngyp]

    ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
    ax1.plot(r_dat, p_bk, ls='-', linewidth=2, color='red', label='Background model')
    ax1.plot(r_dat, p_tot, ls='-', linewidth=2, color='grey', label='Total model')
    for ips in range(len(p_ps)):
        if ips == 0:
            ax1.plot(r_dat, p_ps[ips], ls='-', linewidth=2, color='green', label='Point source model')
        else:
            ax1.plot(r_dat, p_ps[ips], ls='-', linewidth=2, color='green')
    ax1.errorbar(r_dat, p_dat, yerr=err_tot,
                 marker='o', elinewidth=2, color='k',
                 markeredgecolor="black", markerfacecolor="k",
                 ls ='', label='Data')
    ax1.set_ylabel('Profile (deg$^{-2}$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_xticks([])
    ax1.legend()
    
    # Add extra unit axes
    ax2 = ax1.twinx()
    ax2.plot(r_dat, (p_cl*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
    ax2.set_ylabel('Profile (arcmin$^{-2}$)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                 (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
    
    # Residual plot
    ax3.plot(r_dat, (p_dat-p_tot)/err_tot,linestyle='', marker='o', color='k')
    ax3.plot(r_dat,  r_dat*0, linestyle='-', color='k')
    ax3.plot(r_dat,  r_dat*0+2, linestyle='--', color='k')
    ax3.plot(r_dat,  r_dat*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('Radius (deg)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(outdir+'/MCMC_Profile_bkginc.pdf')
    plt.close()
    
    #========== Plot 5: profile, background subtracted, energy bins
    pdf_pages = PdfPages(outdir+'/MCMC_ProfileSlice_bkgsub.pdf')
    
    for i in range(len(Ebins)):
        Ebinprint = '{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)

        cntmap_data = data[i,:,:]*corner_mask[i,:,:]
        cntmap_tot  = modbest['total'][i,:,:]*corner_mask[i,:,:]
        cntmap_cl   = modbest['cluster'][i,:,:]*corner_mask[i,:,:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_dat, p_dat, err_dat = map_tools.radial_profile_cts(cntmap_data,
                                                                 [coord.icrs.ra.to_value('deg'),
                                                                  coord.icrs.dec.to_value('deg')],
                                                                 stddev=np.sqrt(cntmap_tot),
                                                                 header=header,
                                                                 binsize=profile_reso.to_value('deg'),
                                                                 stat='POISSON', counts2brightness=True,
                                                                 residual=True)
            r_tot, p_tot, err_tot = map_tools.radial_profile_cts(cntmap_tot,
                                                                 [coord.icrs.ra.to_value('deg'),
                                                                  coord.icrs.dec.to_value('deg')],
                                                                 stddev=np.sqrt(cntmap_tot),
                                                                 header=header,
                                                                 binsize=profile_reso.to_value('deg'),
                                                                 stat='POISSON', counts2brightness=True,
                                                                 residual=True)
            r_cl, p_cl, err_cl = map_tools.radial_profile_cts(cntmap_cl,
                                                              [coord.icrs.ra.to_value('deg'),
                                                               coord.icrs.dec.to_value('deg')],
                                                              stddev=np.sqrt(cntmap_cl),
                                                              header=header,
                                                              binsize=profile_reso.to_value('deg'),
                                                              stat='POISSON', counts2brightness=True,
                                                              residual=True)
        
        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1])
        
        xlim = [np.amin(r_dat)/1.1, np.amax(r_dat)*1.1]
        rngyp = 1.2*np.nanmax(p_dat-(p_tot-p_cl)+3*err_dat)
        rngym = 0.5*np.nanmin((p_dat-(p_tot-p_cl))[p_dat-(p_tot-p_cl) > 0])
        ylim = [rngym, rngyp]
        
        ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
        ax1.errorbar(r_dat, p_dat-(p_tot-p_cl), yerr=err_tot,
                     marker='o', elinewidth=2, color='k',
                     markeredgecolor="black", markerfacecolor="k",
                     ls ='', label='Data')
        ax1.set_ylabel('Profile (deg$^{-2}$)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_xticks([])
        ax1.legend()
        ax1.set_title('E=['+Ebinprint+'] GeV')

        # Add extra unit axes
        ax2 = ax1.twinx()
        ax2.plot(r_dat, (p_cl*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
        ax2.set_ylabel('Profile (arcmin$^{-2}$)')
        ax2.set_yscale('log')
        ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                     (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
        
        # Residual plot
        ax3.plot(r_dat, (p_dat-p_tot)/err_tot, linestyle='', marker='o', color='k')
        ax3.plot(r_dat,  r_dat*0, linestyle='-', color='k')
        ax3.plot(r_dat,  r_dat*0+2, linestyle='--', color='k')
        ax3.plot(r_dat,  r_dat*0-2, linestyle='--', color='k')
        ax3.set_xlim(xlim[0], xlim[1])
        ax3.set_ylim(-5, 5)
        ax3.set_xlabel('Radius (deg)')
        ax3.set_ylabel('$\\chi$')
        ax3.set_xscale('log')
        
        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()

    #========== Plot 6: profile, background included, energy bins
    pdf_pages = PdfPages(outdir+'/MCMC_ProfileSlice_bkginc.pdf')
    
    for i in range(len(Ebins)):
        Ebinprint = '{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)

        cntmap_data = data[i,:,:]*corner_mask[i,:,:]
        cntmap_tot = modbest['total'][i,:,:]*corner_mask[i,:,:]
        cntmap_cl  = modbest['cluster'][i,:,:]*corner_mask[i,:,:]
        cntmap_bk  = modbest['background'][i,:,:]*corner_mask[i,:,:]
        cntmap_ps  = []
        for ips in range(len(modbest['point_sources'])):
            cntmap_ps.append((modbest['point_sources'][ips])[i,:,:]*corner_mask[i,:,:])
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_dat, p_dat, err_dat = map_tools.radial_profile_cts(cntmap_data,
                                                                 [coord.icrs.ra.to_value('deg'),
                                                                  coord.icrs.dec.to_value('deg')],
                                                                 stddev=np.sqrt(cntmap_tot),
                                                                 header=header,
                                                                 binsize=profile_reso.to_value('deg'),
                                                                 stat='POISSON', counts2brightness=True,
                                                                 residual=True)
            r_tot, p_tot, err_tot = map_tools.radial_profile_cts(cntmap_tot,
                                                                 [coord.icrs.ra.to_value('deg'),
                                                                  coord.icrs.dec.to_value('deg')],
                                                                 stddev=np.sqrt(cntmap_tot),
                                                                 header=header,
                                                                 binsize=profile_reso.to_value('deg'),
                                                                 stat='POISSON', counts2brightness=True,
                                                                 residual=True)
            r_cl, p_cl, err_cl = map_tools.radial_profile_cts(cntmap_cl,
                                                              [coord.icrs.ra.to_value('deg'),
                                                               coord.icrs.dec.to_value('deg')],
                                                              stddev=np.sqrt(cntmap_tot),
                                                              header=header,
                                                              binsize=profile_reso.to_value('deg'),
                                                              stat='POISSON', counts2brightness=True,
                                                              residual=True)
            r_bk, p_bk, err_bk = map_tools.radial_profile_cts(cntmap_bk,
                                                              [coord.icrs.ra.to_value('deg'),
                                                               coord.icrs.dec.to_value('deg')],
                                                              stddev=np.sqrt(cntmap_tot),
                                                              header=header,
                                                              binsize=profile_reso.to_value('deg'),
                                                              stat='POISSON', counts2brightness=True,
                                                              residual=True)
            
            p_ps = []
            for ips in range(len(modbest['point_sources'])):
                r_psi, p_psi, err_psi = map_tools.radial_profile_cts(cntmap_ps[ips],
                                                                     [coord.icrs.ra.to_value('deg'),
                                                                      coord.icrs.dec.to_value('deg')],
                                                                     stddev=np.sqrt(cntmap_tot),
                                                                     header=header,
                                                                     binsize=profile_reso.to_value('deg'),
                                                                     stat='POISSON', counts2brightness=True,
                                                                     residual=True)
                p_ps.append(p_psi)

        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1])
        
        xlim = [np.amin(r_dat)/1.1, np.amax(r_dat)*1.1]
        rngyp = 1.2*np.nanmax(p_dat+3*err_tot)
        rngym = 0.5*np.nanmin(p_dat[p_dat > 0])
        ylim = [rngym, rngyp]
        
        ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
        ax1.plot(r_dat, p_bk, ls='-', linewidth=2, color='red', label='Background model')
        for ips in range(len(p_ps)):
            if ips == 0:
                ax1.plot(r_dat, p_ps[ips], ls='-', linewidth=2, color='green', label='Point source model')
            else:
                ax1.plot(r_dat, p_ps[ips], ls='-', linewidth=2, color='green')
        ax1.plot(r_dat, p_tot, ls='-', linewidth=2, color='grey', label='Total model')
        ax1.errorbar(r_dat, p_dat, yerr=err_dat,
                     marker='o', elinewidth=2, color='k',
                     markeredgecolor="black", markerfacecolor="k",
                     ls ='', label='Data')
        ax1.set_ylabel('Profile (deg$^{-2}$)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_xticks([])
        ax1.legend()
        ax1.set_title('E=['+Ebinprint+'] GeV')

        # Add extra unit axes
        ax2 = ax1.twinx()
        ax2.plot(r_dat, (p_cl*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
        ax2.set_ylabel('Profile (arcmin$^{-2}$)')
        ax2.set_yscale('log')
        ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                     (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
        
        # Residual plot
        ax3.plot(r_dat, (p_dat-p_tot)/err_tot,linestyle='', marker='o', color='k')
        ax3.plot(r_dat,  r_dat*0, linestyle='-', color='k')
        ax3.plot(r_dat,  r_dat*0+2, linestyle='--', color='k')
        ax3.plot(r_dat,  r_dat*0-2, linestyle='--', color='k')
        ax3.set_xlim(xlim[0], xlim[1])
        ax3.set_ylim(-5, 5)
        ax3.set_xlabel('Radius (deg)')
        ax3.set_ylabel('$\\chi$')
        ax3.set_xscale('log')
        
        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()
    
    #========== Plot 7: spectrum, background included
    #----- Compute a mask
    radmapgrid = np.tile(radmap, (len(Ebins),1,1))
    mask = radmapgrid*0 + 1
    mask[radmapgrid > theta.to_value('deg')] = 0
    mask = mask * corner_mask

    #----- Get the bins
    Emean = 1e-6*(Ebins['E_MIN']+Ebins['E_MAX'])/2
    binsteps = 1e-6*np.append(Ebins['E_MIN'],Ebins['E_MAX'][-1])
    
    #----- Get the model and data
    data_spec       = np.sum(np.sum(mask*data, axis=1), axis=1)
    total_spec      = np.sum(np.sum(mask*modbest['total'], axis=1), axis=1)
    cluster_spec    = np.sum(np.sum(mask*modbest['cluster'], axis=1), axis=1)
    background_spec = np.sum(np.sum(mask*modbest['background'], axis=1), axis=1)
    pointsource_spec = []
    for ips in range(len(modbest['point_sources'])):
        pointsource_spec.append(np.sum(np.sum(mask*modbest['point_sources'][ips], axis=1), axis=1))
    
    #----- Get the MC
    Nmc = MC_model['cluster'].shape[0]
    cluster_mc_spec    = np.zeros((Nmc, len(Ebins)))
    for i in range(Nmc):
        cluster_mci_spec    = np.sum(np.sum(mask*MC_model['cluster'][i,:,:,:], axis=1), axis=1)
        cluster_mc_spec[i, :]    = cluster_mci_spec
    cluster_up_spec    = np.percentile(cluster_mc_spec, (100-conf)/2.0, axis=0)
    cluster_lo_spec    = np.percentile(cluster_mc_spec, 100 - (100-conf)/2.0, axis=0)

    #----- Figure
    fig = plt.figure(1, figsize=(8, 6))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(Emean, data_spec, yerr=np.sqrt(total_spec),
                 fmt='ko',capsize=0, linewidth=2, zorder=2, label='Data')
    plt.step(binsteps, np.append(cluster_spec,cluster_spec[-1]),
             where='post', color='blue', linewidth=2, label='Cluster model')
    plt.step(binsteps, np.append(background_spec, background_spec[-1]),
             where='post', color='red', linewidth=2, label='Background model')
    plt.step(binsteps, np.append(total_spec, total_spec[-1]),
             where='post', color='grey', linewidth=2, label='Total model')
    for ips in range(len(pointsource_spec)):
        if ips == 0:
            plt.step(binsteps, np.append(pointsource_spec[ips], (pointsource_spec[ips])[-1]),
                     where='post', color='green', linewidth=2, label='Point source')
        else:
            plt.step(binsteps, np.append(pointsource_spec[ips], (pointsource_spec[ips])[-1]),
                     where='post', color='green', linewidth=2)
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    ax = plt.gca()
    plt.ylim(np.amax(np.array([0.08, ax.get_ylim()[0]])), ax.get_ylim()[1])
    ax.set_xticklabels([])
    plt.legend()
    plt.title('Spectrum within $\\theta = $'+str(theta))

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.plot(Emean, (data_spec-total_spec)/np.sqrt(total_spec),
             marker='o', color='k', linestyle='')
    plt.axhline(0, color='0.5', linestyle='-')
    plt.axhline(-3, color='0.5', linestyle='--')
    plt.axhline(+3, color='0.5', linestyle='--')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Residual ($\\Delta N /\sqrt{N}$)')
    plt.xscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    plt.ylim(-5, 5)

    plt.savefig(outdir+'/MCMC_SpectrumResidualHist.pdf')
    plt.close()

    #========== Plot 8: spectrum, background subtracted
    fig = plt.figure(1, figsize=(8, 6))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(Emean, data_spec-(total_spec-cluster_spec), yerr=np.sqrt(total_spec),
                 xerr=[Emean-1e-6*Ebins['E_MIN'], 1e-6*Ebins['E_MAX']-Emean],fmt='ko',
                 capsize=0, linewidth=2, zorder=2, label='Data')
    plt.plot(Emean, cluster_spec, color='blue', linewidth=2, label='Best-fit cluster model')
    plt.plot(Emean, cluster_up_spec, color='blue', linewidth=1, linestyle='--', label=str(conf)+'% C.L.')
    plt.plot(Emean, cluster_lo_spec, color='blue', linewidth=1, linestyle='--')
    plt.fill_between(Emean, cluster_up_spec, cluster_lo_spec, alpha=0.2, color='blue')
    for i in range(Nmc):
        plt.plot(Emean, cluster_mc_spec[i, :], color='blue', linewidth=1, alpha=0.05)
    plt.ylabel('Background subtracted counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    ax = plt.gca()
    plt.ylim(np.amax(np.array([0.08, ax.get_ylim()[0]])), ax.get_ylim()[1])
    plt.ylim(1e-1, 1e4)                       #                  <<<<<<<------------------>>>>>>
    ax.set_xticklabels([])
    plt.legend()
    plt.title('Spectrum within $\\theta = $'+str(theta))

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.plot(Emean, (data_spec-total_spec)/np.sqrt(total_spec),
             marker='o', color='k', linestyle='')
    plt.axhline(0, color='0.5', linestyle='-')
    plt.axhline(-3, color='0.5', linestyle='--')
    plt.axhline(+3, color='0.5', linestyle='--')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Residual ($\\Delta N /\sqrt{N}$)')
    plt.xscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    plt.ylim(-5, 5)

    plt.savefig(outdir+'/MCMC_SpectrumResidual.pdf')
    plt.close()

    
#==================================================
# Read the data
#==================================================

def read_data(input_files):
    """
    Read the data to extract the necessary information
    
    Parameters
    ----------
    - input_files (str list): file where the data is stored
    and file where the grid model is stored

    Output
    ------
    - data (ndarray): table containing the data
    - modgrid (dict): grid model to be interpolated

    """
    
    # Get measured data
    hdu = fits.open(input_files[0])
    data = hdu[0].data
    header = hdu[0].header
    Ebins = hdu[2].data
    hdu.close()
    
    # Get expected    
    hdul = fits.open(input_files[1])
    Nps = len(hdul)-7 # 7=1(primary)+3(cl)+2(bk)+1(ps index)
    
    sample_spa    = hdul[1].data
    sample_spe    = hdul[2].data
    models_cl     = hdul[3].data
    bk_sample_spe = hdul[4].data
    models_bk     = hdul[5].data
    ps_sample_spe = hdul[6].data
    models_ps_list = []
    for ips in range(Nps):
        models_ps_list.append(hdul[7+ips].data)
    hdu.close()

    gridshape = models_cl.shape
    
    # Check that the grid is the same, as expected
    if data.shape != models_cl[0,0,:,:,:].shape:
        print('!!!!! WARNING: it is possible that we have a problem with the grid !!!!!')
        
    # Extract and fill model_grid
    x_val = np.linspace(0, gridshape[4]-1, gridshape[4])  # pixel 1d value along RA
    y_val = np.linspace(0, gridshape[3]-1, gridshape[3])  # pixel 1d value along Dec
    e_val = np.linspace(0, gridshape[2]-1, gridshape[2])  # pixel 1d value along energy

    ee_val, yy_val, xx_val = np.meshgrid(e_val, y_val, x_val, indexing='ij') # 3D gids
    
    xxf_val = xx_val.flatten() # 3D grid flattened
    yyf_val = yy_val.flatten()
    eef_val = ee_val.flatten()
    
    modgrid = {'header':header,
               'Ebins':Ebins,
               'x_val':x_val, # x pixel indexing
               'y_val':y_val, # y pixel indexing
               'e_val':e_val, # e pixel indexing
               'spe_val':sample_spe['spectral_val'],
               'spa_val':sample_spa['spatial_val'],
               'bk_spe_val':bk_sample_spe['bk_spectral_val'],
               'ps_spe_val':ps_sample_spe['ps_spectral_val'],
               'xx_val':xx_val, # x pixel 3d grid
               'yy_val':yy_val, # y pixel 3d grid
               'ee_val':ee_val, # e pixel 3d grid
               'xxf_val':xxf_val, # x pixel 3d grid flattened
               'yyf_val':yyf_val, # y pixel 3d grid flattened
               'eef_val':eef_val, # e pixel 3d grid flattened
               'models_cl':models_cl,
               'models_bk':models_bk,
               'models_ps_list':models_ps_list}
    
    return data, modgrid


#==================================================
# Build a circular mask that masks the corner
#==================================================

def get_corner_mask(input_files):
    """
    Build a circular mask that masks the corner
    
    Parameters
    ----------
    - input_files (str list): file where the data is stored
    and file where the grid model is stored

    Output
    ------
    - corner_mask (ndarray): 0=mask, 1=unmasked

    """
    
    # Get measured data
    hdu = fits.open(input_files[0])
    data = hdu[0].data
    header = hdu[0].header
    Ebins = hdu[2].data
    hdu.close()
    del header['NAXIS3']
    header['NAXIS'] = 2

    # Build the mask
    ra_map, dec_map = map_tools.get_radec_map(header)
    distmap = map_tools.greatcircle(ra_map, dec_map, np.mean(ra_map), np.mean(dec_map))

    corner_mask = distmap*0+1
    corner_mask[distmap > 0.5*(np.amax(dec_map)-np.amin(dec_map))] = 0

    # Replicate the mask along E
    corner_mask = np.tile(corner_mask, (len(Ebins), 1,1))

    return corner_mask


#==================================================
# MCMC: Defines log prior
#==================================================

def lnprior(params, par_min, par_max, g_prior):
    '''
    Return the flat prior on parameters

    Parameters
    ----------
    - params (list): the parameters
    - par_min (list): the minimum value for params
    - par_max (list): the maximum value for params
    - g_prior (tupple array): gaussian prior as [(mu,sigma),(mu,sigma),...]

    Output
    ------
    - prior (float): the value of the prior, either 0 or -inf

    '''

    prior = 0.0
    
    for i in range(len(params)):
        if not (np.isnan(g_prior[i][0]) or np.isnan(g_prior[i][1])):
            prior += -0.5*(params[i]-g_prior[i][0])**2 / g_prior[i][1]**2
    
    for i in range(len(params)):
        if params[i] <= par_min[i] or params[i] >= par_max[i] :
            prior = -np.inf
            
    return prior


#==================================================
# MCMC: Defines log likelihood
#==================================================

def lnlike(params, data, modgrid, par_min, par_max, g_prior, mask, gauss=False):
    '''
    Return the log likelihood for the given parameters

    Parameters
    ----------
    - params (list): the parameters
    - data (Table): the data flux and errors
    - modgrid (Table): grid of model for different scaling to be interpolated
    - par_min (list): the minimum value for params
    - par_max (list): the maximum value for params
    - g_prior (tupple array): gaussian prior as [(mu,sigma),(mu,sigma),...]
    - gauss (bool): use a gaussian approximation for errors

    Output
    ------
    - lnlike (float): the value of the log likelihood
    '''
    
    #---------- Get the prior
    prior = lnprior(params, par_min, par_max, g_prior)
    if prior == -np.inf: # Should not go for the model if prior out
        return -np.inf
    if np.isinf(prior):
        return -np.inf
    if np.isnan(prior):
        return -np.inf
    
    #---------- Get the test model
    if params[0] <= 0: # should never happen, but debug in case it does
        import pdb
        pdb.set_trace()

    test_model = model_specimg(modgrid, params)
    
    #---------- Compute the Gaussian likelihood
    # Gaussian likelihood
    if gauss:
        chi2 = (data - test_model['total'])**2/np.sqrt(test_model['total'])**2
        lnL = -0.5*np.nansum(chi2*mask)

    # Poisson with Bkg
    else:        
        L_i1 = test_model['total']
        L_i2 = data*np.log(test_model['total'])
        lnL  = -np.nansum((L_i1 - L_i2)*mask)
    
    # In case of NaN, goes to infinity
    if np.isnan(lnL):
        lnL = -np.inf
        
    return lnL + prior


#==================================================
# MCMC: Defines model
#==================================================

def model_specimg(modgrid, params):
    '''
    Gamma ray model for the MCMC

    Parameters
    ----------
    - modgrid (array): grid of models
    - param (list): the parameter to sample in the model

    Output
    ------
    - output_model (array): the output model in units of the input expected
    '''

    # 0, 1, 2 -> cluster
    # 3, 4    -> Norm, spec bkg
    # 5, 6    -> Norm, spec ps1
    # 7, 8    -> Norm, spec ps2
    # ...

    # Cluster component
    outf_cl = interpn((modgrid['spa_val'], modgrid['spe_val'],
                       modgrid['e_val'], modgrid['y_val'], modgrid['x_val']),
                      modgrid['models_cl'],
                      (params[1], params[2], modgrid['eef_val'], modgrid['yyf_val'], modgrid['xxf_val']),
                      method='linear')
    out_cl = np.reshape(outf_cl, modgrid['xx_val'].shape)
    out_cl = params[0]*out_cl
    
    # Background component
    f_bk = interp1d(modgrid['bk_spe_val'], modgrid['models_bk'], axis=0)
    out_bk = params[3] * f_bk(params[4])
    
    # Point sources component
    out_ps = []
    for ips in range(len(modgrid['models_ps_list'])):
        f_ps = interp1d(modgrid['ps_spe_val'], modgrid['models_ps_list'][ips], axis=0)
        out_ps.append(params[5+ips*2] * f_ps(params[6+ips*2]))
        
    # Add normalization parameter and save
    tot_ps = out_cl*0
    for ips in range(len(modgrid['models_ps_list'])):
        tot_ps = tot_ps + out_ps[ips]
    total = out_cl + out_bk + tot_ps

    output_model = {'cluster':out_cl, 'background':out_bk, 'point_sources':out_ps, 'total':total}

    return output_model


#==================================================
# iminuit fit
#==================================================

def iminuit_fit(data, modgrid, parname, par0, par_min, par_max, filesave):
    '''
    Fit the model to the data using iminuit

    Parameters
    ----------

    Output
    ------
    '''
    try:
        from iminuit import Minuit
    
        def iminuit_lnlike(params):
            test_model = model_specimg(modgrid, params)
            L_i1 = test_model['total']
            L_i2 = data*np.log(test_model['total'])
            return np.nansum((L_i1 - L_i2))
        
        m = Minuit(iminuit_lnlike, par0, name=parname)
        m.errordef = Minuit.LIKELIHOOD
        
        # Def limnits
        limits = []
        for ip in range(len(par_min)):
            limits.append((par_min[ip], par_max[ip]))
        m.limits = limits
    
        print(m)
        # run fit
        m.migrad()  # finds minimum of least_squares function
        m.hesse()   # accurately computes uncertainties
        m.minos()   # non symmetric or parabolic errors
        print(m)
    
        orig_stdout = sys.stdout
        f = open(filesave, 'w')
        sys.stdout = f
        print(m)
        sys.stdout = orig_stdout
        f.close()
    except:
        print('')
        print('-----> iminuit fit failed.')

#==================================================
# MCMC: run the fit
#==================================================

def run_constraint(input_files,
                   subdir,
                   nwalkers=10,
                   nsteps=1000,
                   burnin=100,
                   conf=68.0,
                   Nmc=100,
                   prior=None,
                   GaussLike=False,
                   reset_mcmc=False,
                   run_mcmc=True,
                   FWHM=0.1*u.deg,
                   theta=1.0*u.deg,
                   coord=None,
                   profile_reso=0.05*u.deg,
                   app_corner_mask=False,
                   run_minuit=False):
    """
    Run the MCMC spectral imaging constraints
        
    Parameters
    ----------
    - input_file (str): full path to the data and expected model
    - subdir (str): subdirectory of spectral imaging, full path
    - nwalkers (int): number of emcee wlakers
    - nsteps (int): number of emcee MCMC steps
    - burnin (int): number of point to remove assuming it is burnin
    - conf (float): confidence limit percentage for results
    - prior (array of tupple): gaussian prior on parameters
    - Nmc (int): number of monte carlo point when resampling the chains
    - GaussLike (bool): use gaussian approximation of the likelihood
    - reset_mcmc (bool): reset the existing MCMC chains?
    - run_mcmc (bool): run the MCMC sampling?
    - FWHM (quantity): size of the FWHM to be used for smoothing
    - theta (quantity): containment angle for plots
    - coord (SkyCoord): source coordinates for extraction
    - profile_reso (quantity): bin size for profile
    - app_corner_mask (bool): apply a mask to the corner of the map
    - run_minuit (bool): run iminuit before MCMC

    Output
    ------
    The final MCMC chains and plots are saved
    """

    #========== Reset matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    #========== Read the data
    data, modgrid = read_data(input_files)
    if app_corner_mask:
        corner_mask   = get_corner_mask(input_files)
    else:
        corner_mask = data*0+1
    
    #========== Guess parameter definition
    # Normalization, scaling profile \propto profile_input^eta, CRp index
    parname = ['X_{CRp}/X_{CRp,0}', '\\eta_{CRp}', '\\alpha_{CRp}', 'A_{bkg}', '\\Delta \\alpha_{bkg}']    
    par0 = np.array([1.0, np.mean(modgrid['spa_val']), np.mean(modgrid['spe_val']),
                     1.0, np.mean(modgrid['bk_spe_val'])])
    par_min = [0,      np.amin(modgrid['spa_val']), np.amin(modgrid['spe_val']),
               0.0, np.amin(modgrid['bk_spe_val'])]
    par_max = [20.0, np.amax(modgrid['spa_val']), np.amax(modgrid['spe_val']),
               np.inf, np.amax(modgrid['bk_spe_val'])]    

    for i in range(len(modgrid['models_ps_list'])):
        parname.append('A_{ps,'+str(i+1)+'}')
        parname.append('\\Delta \\alpha_{ps,'+str(i+1)+'}')        
        par0 = np.append(par0, 1.0)
        par0 = np.append(par0, np.mean(modgrid['ps_spe_val']))
        par_min.append(0.0)
        par_min.append(np.amin(modgrid['ps_spe_val']))
        par_max.append(np.inf)
        par_max.append(np.amax(modgrid['ps_spe_val']))

    # prior
    if prior is None:
        g_prior = []
        for ip in range(len(parname)):
            g_prior.append((np.nan, np.nan))
    else:
        g_prior = prior

    #========== iMinuit fit
    if run_minuit:
        file_iminuit =  subdir+'/iMinuit_results.txt'
        iminuit_fit(data, modgrid, parname, par0, par_min, par_max, file_iminuit)
    
    #========== Names
    sampler_file1  = subdir+'/MCMC_sampler.pkl'
    sampler_file2  = subdir+'/MCMC_sampler.h5'
    chainstat_file = subdir+'/MCMC_chainstat.txt'
    chainplot_file = subdir+'/MCMC_chainplot'
    
    #========== Start running MCMC definition and sampling    
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(sampler_file2)
    if sampler_exist:
        print('    Existing sampler: '+sampler_file2)
    else:
        print('    No existing sampler found')
        
    #---------- MCMC parameters
    ndim = len(par0)
    if nwalkers < 2*ndim:
        print('nwalkers should be at least twice the number of parameters.')
        print('nwalkers --> '+str(ndim*2))
        print('')
        nwalkers = ndim*2
    
    print('--- MCMC profile parameters: ')
    print('    ndim                = '+str(ndim))
    print('    nwalkers            = '+str(nwalkers))
    print('    nsteps              = '+str(nsteps))
    print('    burnin              = '+str(burnin))
    print('    conf                = '+str(conf))
    print('    reset_mcmc          = '+str(reset_mcmc))
    print('    Gaussian likelihood = '+str(GaussLike))
    print('    Parameter min       = '+str(par_min))
    print('    Parameter max       = '+str(par_max))
    print('    Gaussian prior      = '+str(g_prior))
    print('    Corner mask apps    = '+str(app_corner_mask))

    #---------- Defines the start
    backend = emcee.backends.HDFBackend(sampler_file2)
    pos = mcmc_common.chains_starting_point(par0, 0.1, par_min, par_max, nwalkers)
    if sampler_exist:
        if reset_mcmc:
            print('    Reset MCMC even though sampler already exists')
            backend.reset(nwalkers, ndim)
        else:
            print('    Use existing MCMC sampler')
            print("    --> Initial size: {0}".format(backend.iteration))
            pos = None
    else:
        print('    No pre-existing sampler, start from scratch')
        backend.reset(nwalkers, ndim)
    
    moves = emcee.moves.StretchMove(a=2.0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                    args=[data, modgrid, par_min, par_max, g_prior, corner_mask, GaussLike],
                                    pool=Pool(cpu_count()), moves=moves,
                                    backend=backend)

    #---------- Run the MCMC
    if run_mcmc:
        print('--- Runing '+str(nsteps)+' MCMC steps')
        res = sampler.run_mcmc(pos, nsteps, progress=True)

        # Save the MCMC after the run
        mcmc_common.save_object(sampler, sampler_file1)
        
    #----- Restore chains
    else:
        with open(sampler_file1, 'rb') as f:
            sampler = pickle.load(f)
        
    #---------- Burnin
    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains = sampler.lnprobability[:, burnin:]
    
    #---------- Get the parameter statistics
    par_best, par_percentile = mcmc_common.chains_statistics(param_chains, lnL_chains,
                                                             parname=parname, conf=conf, show=True,
                                                             outfile=chainstat_file)
    
    #---------- Get the well-sampled models
    MC_model   = get_mc_model(modgrid, param_chains, Nmc=Nmc)
    Best_model = model_specimg(modgrid, par_best)

    #---------- Plots and results
    mcmc_common.chains_plots(param_chains, parname, chainplot_file,
                             par_best=par_best, par_percentile=par_percentile, conf=conf,
                             par_min=par_min, par_max=par_max)

    mcmc_common.chains_plots(param_chains[:,:,[0,1,2]], parname[0:3],
                             chainplot_file+'_cl',
                             par_best=par_best[0:3], par_percentile=par_percentile[:,[0,1,2]],
                             conf=conf,
                             par_min=par_min[0:3], par_max=par_max[0:3])

    modelplot(data, Best_model, MC_model, corner_mask, modgrid['header'], modgrid['Ebins'], subdir,
              conf=conf, FWHM=FWHM, theta=theta, coord=coord, profile_reso=profile_reso)
