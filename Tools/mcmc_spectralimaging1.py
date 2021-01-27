"""
This file contains the modules which perform the MCMC spectral imaging constraint. 
"""

#==================================================
# Requested imports
#==================================================

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
                     includeIC=False,
                     rm_tmp=False):
    """
    Build a grid of models for the cluster and background. The 
    background model is obtained using the likelihood best fit
    according to the given tested cluster model.
    
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
    - includeIC (bool): include inverse Compton in the model
    - rm_tmp (bool): remove temporary files

    Output
    ------
    - All temporary file (e.g. cluster templates, likelihood fit models, ...)
    and the Grid_Sampling.fits file
    
    """
    
    # Save the cluster model before modification
    cluster_tmp = copy.deepcopy(cpipe.cluster)
    
    #===== Loop over all models to be tested
    spatial_npt = len(spatial_value)
    spectral_npt = len(spectral_value)
    
    for imod in range(spatial_npt):
        for jmod in range(spectral_npt):
            print('--- Building model template '+str(1+jmod+imod*spectral_npt)+'/'+str(spatial_npt*spectral_npt))
            
            #---------- Indexing
            spatial_i = spatial_value[imod]            
            spectral_j   = spectral_value[jmod]
            extij = 'TMP_'+str(imod)+'_'+str(jmod)
            
            #---------- Compute the model spectrum, map, and xml model file
            # Re-scaling        
            cluster_tmp.density_crp_model  = {'name':'User',
                                              'radius':rad, 'profile':prof_ini.value ** spatial_i}
            cluster_tmp.spectrum_crp_model = {'name':'PowerLaw', 'Index':spectral_j}
            
            # Cluster model
            make_cluster_template.make_map(cluster_tmp,
                                           subdir+'/Model_Map_'+extij+'.fits',
                                           Egmin=cpipe.obs_setup.get_emin(),Egmax=cpipe.obs_setup.get_emax(),
                                           includeIC=includeIC)
            make_cluster_template.make_spectrum(cluster_tmp,
                                                subdir+'/Model_Spectrum_'+extij+'.txt',
                                                energy=np.logspace(-1,5,1000)*u.GeV,
                                                includeIC=includeIC)
            
            # xml model
            model_tot = gammalib.GModels(cpipe.output_dir+'/Ana_Model_Input_Stack.xml')
            clencounter = 0
            for i in range(len(model_tot)):
                if model_tot[i].name() == cluster_tmp.name:
                    spefn = subdir+'/Model_Spectrum_'+extij+'.txt'
                    model_tot[i].spectral().filename(spefn)
                    spafn = subdir+'/Model_Map_'+extij+'.fits'
                    model_tot[i].spatial(gammalib.GModelSpatialDiffuseMap(spafn))
                    clencounter += 1
            if clencounter != 1:
                raise ValueError('No cluster encountered in the input stack model')
            model_tot.save(subdir+'/Model_Input_'+extij+'.xml')

            #---------- Likelihood fit
            like = ctools.ctlike()
            like['inobs']           = cpipe.output_dir+'/Ana_Countscube.fits'
            like['inmodel']         = subdir+'/Model_Input_'+extij+'.xml'
            like['expcube']         = cpipe.output_dir+'/Ana_Expcube.fits'
            like['psfcube']         = cpipe.output_dir+'/Ana_Psfcube.fits'
            like['bkgcube']         = cpipe.output_dir+'/Ana_Bkgcube.fits'
            like['edispcube']       = cpipe.output_dir+'/Ana_Edispcube.fits'
            like['edisp']           = cpipe.spec_edisp
            like['outmodel']        = subdir+'/Model_Output_'+extij+'.xml'
            like['outcovmat']       = 'NONE'
            like['statistic']       = cpipe.method_stat
            like['refit']           = False
            like['like_accuracy']   = 0.005
            like['max_iter']        = 50
            like['fix_spat_for_ts'] = False
            like['logfile']         = subdir+'/Model_Output_log_'+extij+'.txt'
            like.logFileOpen()
            like.execute()
            like.logFileClose()

            #---------- Compute the 3D residual cube
            cpipe._rm_source_xml(subdir+'/Model_Output_'+extij+'.xml',
                                subdir+'/Model_Output_Cluster_'+extij+'.xml',
                                cluster_tmp.name)
            
            modcube = cubemaking.model_cube(cpipe.output_dir,
                                            cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                            cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                            cpipe.spec_ebinalg,
                                            edisp=cpipe.spec_edisp,
                                            stack=cpipe.method_stack,
                                            silent=True,
                                            logfile=subdir+'/Model_Cube_log_'+extij+'.txt',
                                            inmodel_usr=subdir+'/Model_Output_'+extij+'.xml',
                                            outmap_usr=subdir+'/Model_Cube_'+extij+'.fits')
            
            modcube_Cl = cubemaking.model_cube(cpipe.output_dir,
                                               cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                               cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                               cpipe.spec_ebinalg,
                                               edisp=cpipe.spec_edisp,
                                               stack=cpipe.method_stack, silent=True,
                                               logfile=subdir+'/Model_Cube_Cluster_log_'+extij+'.txt',
                                               inmodel_usr=subdir+'/Model_Output_Cluster_'+extij+'.xml',
                                               outmap_usr=subdir+'/Model_Cube_Cluster_'+extij+'.fits')

    #===== Build the grid
    hdul = fits.open(subdir+'/Model_Cube_TMP_'+str(0)+'_'+str(0)+'.fits')
    cnt0 = hdul[0].data
    hdul.close()
    modgrid_bk = np.zeros((spatial_npt, spectral_npt,
                           cnt0.shape[0], cnt0.shape[1], cnt0.shape[2]))
    modgrid_cl = np.zeros((spatial_npt, spectral_npt,
                           cnt0.shape[0], cnt0.shape[1], cnt0.shape[2]))
    
    for imod in range(spatial_npt):
        for jmod in range(spectral_npt):
            extij = 'TMP_'+str(imod)+'_'+str(jmod)
            
            hdul1 = fits.open(subdir+'/Model_Cube_'+extij+'.fits')
            hdul2 = fits.open(subdir+'/Model_Cube_Cluster_'+extij+'.fits')
            modgrid_bk[imod,jmod,:,:,:] = hdul2[0].data
            modgrid_cl[imod,jmod,:,:,:] = hdul1[0].data - hdul2[0].data
            hdul1.close()
            hdul2.close()
            
    #===== Save in a table
    scal_spa = Table()
    scal_spa['spatial_idx'] = spatial_idx
    scal_spa['spatial_val'] = spatial_value
    scal_spa_hdu = fits.BinTableHDU(scal_spa)
    
    scal_spe = Table()        
    scal_spe['spectral_idx'] = spectral_idx
    scal_spe['spectral_val'] = spectral_value
    scal_spe_hdu = fits.BinTableHDU(scal_spe)
    
    grid_cl_hdu = fits.ImageHDU(modgrid_cl)
    grid_bk_hdu = fits.ImageHDU(modgrid_bk)
    
    hdul = fits.HDUList()
    hdul.append(scal_spa_hdu)
    hdul.append(scal_spe_hdu)
    hdul.append(grid_bk_hdu)
    hdul.append(grid_cl_hdu)
    hdul.writeto(subdir+'/Grid_Sampling.fits', overwrite=True)

    #===== Save the properties of the last computation run
    np.save(subdir+'/Grid_Parameters.npy',
            [cpipe.cluster, spatial_value, spectral_value], allow_pickle=True)
    
    #===== remove TMP files
    if rm_tmp:
        for imod in range(spatial_scaling_npt):
            for jmod in range(spectral_slope_npt):
                extij = 'TMP_'+str(imod)+'_'+str(jmod)
                os.remove(subdir+'/Model_Map_'+extij+'.fits')
                os.remove(subdir+'/Model_Spectrum_'+extij+'.txt')
                os.remove(subdir+'/Model_Cube_'+extij+'.fits')
                os.remove(subdir+'/Model_Cube_log_'+extij+'.txt')
                os.remove(subdir+'/Model_Cube_Cluster_'+extij+'.fits')
                os.remove(subdir+'/Model_Cube_Cluster_log_'+extij+'.txt')
                os.remove(subdir+'/Model_Input_'+extij+'.xml')
                os.remove(subdir+'/Model_Output_'+extij+'.xml')
                os.remove(subdir+'/Model_Output_log_'+extij+'.txt')
                os.remove(subdir+'/Model_Output_Cluster_'+extij+'.xml')


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
    
    MC_model_background = np.zeros((Nmc, modgrid['xx_val'].shape[0],
                                    modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2]))
    MC_model_cluster = np.zeros((Nmc, modgrid['xx_val'].shape[0],
                                 modgrid['xx_val'].shape[1], modgrid['xx_val'].shape[2]))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        mods = model_specimg(modgrid, param_MC)
        MC_model_cluster[i,:,:,:]    = mods['cluster']
        MC_model_background[i,:,:,:] = mods['background']

    MC_models = {'cluster':MC_model_cluster,
                 'background':MC_model_background}
    
    return MC_models


#==================================================
# Plot the output fit model
#==================================================

def modelplot(data, modbest, MC_model, header, Ebins, outdir,
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
    fig = plt.figure(0, figsize=(18, 4))
    ax = plt.subplot(131, projection=proj)
    plt.imshow(gaussian_filter(np.sum(data, axis=0), sigma=sigma_sm),
                origin='lower', cmap='magma',norm=SymLogNorm(1, base=10))
    cb = plt.colorbar()
    plt.title('Data (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    
    ax = plt.subplot(132, projection=proj)
    plt.imshow(gaussian_filter(np.sum(modbest['cluster']+modbest['background'],axis=0), sigma=sigma_sm),
               origin='lower', cmap='magma', norm=SymLogNorm(1, base=10, vmin=cb.norm.vmin, vmax=cb.norm.vmax))
    plt.colorbar()
    plt.title('Model (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    vmin0 = np.amin(gaussian_filter(np.sum(data-(modbest['cluster']+modbest['background']), axis=0),
                                    sigma=sigma_sm))
    vmax0 = np.amax(gaussian_filter(np.sum(data-(modbest['cluster']+modbest['background']), axis=0),
                                    sigma=sigma_sm))
    vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])   
    ax = plt.subplot(133, projection=proj)
    plt.imshow(gaussian_filter(np.sum(data-(modbest['cluster']+modbest['background']), axis=0), sigma=sigma_sm),
               origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
    plt.colorbar()
    filt1 = gaussian_filter(np.sum(data-(modbest['cluster']+modbest['background']), axis=0), sigma=sigma_sm)
    filt2 = np.sqrt(gaussian_filter(np.sum((modbest['cluster']+modbest['background']), axis=0), sigma=sigma_sm))
    snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2
    cont = ax.contour(snr, levels=np.array([-9,-7,-5,-3,3,5,7,9]), colors='black')
    plt.title('Residual (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    
    plt.savefig(outdir+'/MCMC_MapResidual.pdf')
    plt.close()

    #========== Plot 2: map, Data - model, energy bins
    pdf_pages = PdfPages(outdir+'/MCMC_MapSliceResidual.pdf')
    
    for i in range(len(Ebins)):
        Ebinprint = '{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)
    
        fig = plt.figure(0, figsize=(18, 4))
        ax = plt.subplot(131, projection=proj)
        plt.imshow(gaussian_filter(data[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='magma', norm=SymLogNorm(1, base=10))
        cb = plt.colorbar()
        plt.title('Data (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        
        ax = plt.subplot(132, projection=proj)
        plt.imshow(gaussian_filter((modbest['cluster']+modbest['background'])[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='magma', norm=SymLogNorm(1, base=10,vmin=cb.norm.vmin, vmax=cb.norm.vmax))
        plt.colorbar()
        plt.title('Model (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')

        vmin0 = np.amin(gaussian_filter((data-(modbest['cluster']+modbest['background']))[i,:,:], sigma=sigma_sm))
        vmax0 = np.amax(gaussian_filter((data-(modbest['cluster']+modbest['background']))[i,:,:], sigma=sigma_sm))
        vmm = np.amax([np.abs(vmin0), np.abs(vmax0)])
        ax = plt.subplot(133, projection=proj)
        plt.imshow(gaussian_filter((data-(modbest['cluster']+modbest['background']))[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='RdBu', vmin=-vmm, vmax=vmm)
        filt1 = gaussian_filter((data-(modbest['cluster']+modbest['background']))[i,:,:], sigma=sigma_sm)
        filt2 = np.sqrt(gaussian_filter((modbest['cluster']+modbest['background'])[i,:,:], sigma=sigma_sm))
        snr   = 2*sigma_sm*np.sqrt(np.pi) * filt1 / filt2
        cont = ax.contour(snr, levels=np.array([-9,-7,-5,-3,3,5,7,9]), colors='black')
        plt.colorbar()
        plt.title('Residual (counts) - E=['+Ebinprint+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')

        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()

    #========== Plot 3: profile, background subtracted, stack
    cntmap_data = np.sum(data, axis=0)
    cntmap_tot  = np.sum(modbest['background']+modbest['cluster'], axis=0)
    cntmap_cl   = np.sum(modbest['cluster'], axis=0)

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
            r_cl_i, p_cl_i, err_cl_i = map_tools.radial_profile_cts(np.sum(MC_model['cluster'][i,:,:,:],axis=0),
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

    xlim = [np.amin(r_dat)/2.0, np.amax(r_dat)*2.0]
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
    ax1.fill_between(r_dat, p_cl_up, p_cl_lo, alpha=0.3, color='blue')
    for i in range(Nmc):
        ax1.plot(r_dat, p_cl_mc[i, :], color='blue', linewidth=1, alpha=0.1)    
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
    
    fig.savefig(outdir+'/MCMC_Profile_bkgsub.pdf')
    plt.close()

    #========== Plot 4: profile, background included, stack
    cntmap_data = np.sum(data, axis=0)
    cntmap_tot  = np.sum(modbest['cluster']+modbest['background'], axis=0)
    cntmap_cl   = np.sum(modbest['cluster'], axis=0)
    cntmap_bk   = np.sum(modbest['background'], axis=0)
    
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
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(r_dat)/2.0, np.amax(r_dat)*2.0]
    rngyp = 1.2*np.nanmax(p_dat+3*err_tot)
    rngym = 0.5*np.nanmin((p_dat)[p_dat > 0])
    ylim = [rngym, rngyp]

    ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
    ax1.plot(r_dat, p_bk, ls='-', linewidth=2, color='red', label='Background model')
    ax1.plot(r_dat, p_tot, ls='-', linewidth=2, color='grey', label='Total model')
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

        cntmap_data = data[i,:,:]
        cntmap_tot  = modbest['background'][i,:,:]+modbest['cluster'][i,:,:]
        cntmap_cl   = modbest['cluster'][i,:,:]

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
        
        xlim = [np.amin(r_dat)/2.0, np.amax(r_dat)*2.0]
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

        cntmap_data = data[i,:,:]
        cntmap_tot = modbest['cluster'][i,:,:] + modbest['background'][i,:,:]
        cntmap_cl  = modbest['cluster'][i,:,:]
        cntmap_bk  = modbest['background'][i,:,:]
    
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
        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1])
        
        xlim = [np.amin(r_dat)/2.0, np.amax(r_dat)*2.0]
        rngyp = 1.2*np.nanmax(p_dat+3*err_tot)
        rngym = 0.5*np.nanmin(p_dat[p_dat > 0])
        ylim = [rngym, rngyp]
        
        ax1.plot(r_dat, p_cl, ls='-', linewidth=2, color='blue', label='Cluster model')
        ax1.plot(r_dat, p_bk, ls='-', linewidth=2, color='red', label='Background model')
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

    #----- Get the bins
    Emean = 1e-6*(Ebins['E_MIN']+Ebins['E_MAX'])/2
    binsteps = 1e-6*np.append(Ebins['E_MIN'],Ebins['E_MAX'][-1])

    #----- Get the model and data
    data_spec       = np.sum(np.sum(mask*data, axis=1), axis=1)
    cluster_spec    = np.sum(np.sum(mask*modbest['cluster'], axis=1), axis=1)
    background_spec = np.sum(np.sum(mask*modbest['background'], axis=1), axis=1)
    
    #----- Get the MC
    Nmc = MC_model['cluster'].shape[0]
    cluster_mc_spec    = np.zeros((Nmc, len(Ebins)))
    background_mc_spec = np.zeros((Nmc, len(Ebins)))
    tot_mc_spec        = np.zeros((Nmc, len(Ebins)))

    for i in range(Nmc):
        cluster_mci_spec    = np.sum(np.sum(mask*MC_model['cluster'][i,:,:,:], axis=1), axis=1)
        background_mci_spec = np.sum(np.sum(mask*MC_model['background'][i,:,:,:], axis=1), axis=1)
        cluster_mc_spec[i, :]    = cluster_mci_spec
        background_mc_spec[i, :] = background_mci_spec
        tot_mc_spec[i, :]        = background_mci_spec + cluster_mci_spec

    cluster_up_spec    = np.percentile(cluster_mc_spec, (100-conf)/2.0, axis=0)
    cluster_lo_spec    = np.percentile(cluster_mc_spec, 100 - (100-conf)/2.0, axis=0)
    background_up_spec = np.percentile(background_mc_spec, (100-conf)/2.0, axis=0)
    background_lo_spec = np.percentile(background_mc_spec, 100 - (100-conf)/2.0, axis=0)
    tot_up_spec        = np.percentile(tot_mc_spec, (100-conf)/2.0, axis=0)
    tot_lo_spec        = np.percentile(tot_mc_spec, 100 - (100-conf)/2.0, axis=0)

    #========== Plot 8: spectrum, background subtracted
    fig = plt.figure(1, figsize=(8, 6))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(Emean, data_spec, yerr=np.sqrt(cluster_spec+background_spec),
                 fmt='ko',capsize=0, linewidth=2, zorder=2, label='Data')
    plt.step(binsteps, np.append(cluster_spec,cluster_spec[-1]),
             where='post', color='blue', linewidth=2, label='Cluster model')
    plt.step(binsteps, np.append(background_spec, background_spec[-1]),
             where='post', color='red', linewidth=2, label='Background model')
    plt.step(binsteps, np.append(cluster_spec+background_spec, (cluster_spec+background_spec)[-1]),
             where='post', color='grey', linewidth=2, label='Total model')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.legend()
    plt.title('Spectrum within $\\theta = $'+str(theta))

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.plot(Emean, (data_spec-cluster_spec-background_spec)/np.sqrt(cluster_spec+background_spec),
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

    #----- Figure
    fig = plt.figure(1, figsize=(8, 6))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(Emean, data_spec-background_spec, yerr=np.sqrt(cluster_spec+background_spec),
                 xerr=[Emean-1e-6*Ebins['E_MIN'], 1e-6*Ebins['E_MAX']-Emean],fmt='ko',
                 capsize=0, linewidth=2, zorder=2, label='Data')
    plt.plot(Emean, cluster_spec, color='k', linewidth=2, label='Best-fit cluster model')
    plt.plot(Emean, cluster_up_spec, color='blue', linewidth=1, linestyle='--', label=str(conf)+'% C.L.')
    plt.plot(Emean, cluster_lo_spec, color='blue', linewidth=1, linestyle='--')
    plt.fill_between(Emean, cluster_up_spec, cluster_lo_spec, alpha=0.3, color='blue')
    for i in range(Nmc):
        plt.plot(Emean, cluster_mc_spec[i, :], color='blue', linewidth=1, alpha=0.1)
    plt.ylabel('Background subtracted counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.legend()
    plt.title('Spectrum within $\\theta = $'+str(theta))

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.plot(Emean, (data_spec-cluster_spec-background_spec)/np.sqrt(cluster_spec+background_spec),
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
    hdu = fits.open(input_files[1])
    sample_spa = hdu[1].data
    sample_spe = hdu[2].data
    models_bk  = hdu[3].data
    models_cl  = hdu[4].data
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
               'x_val':x_val,
               'y_val':y_val,
               'e_val':e_val,
               'spe_val':sample_spe['spectral_val'],
               'spa_val':sample_spa['spatial_val'],
               'xx_val':xx_val,
               'yy_val':yy_val,
               'ee_val':ee_val,
               'xxf_val':xxf_val,
               'yyf_val':yyf_val,
               'eef_val':eef_val,               
               'models_cl':models_cl,
               'models_bk':models_bk}
    
    return data, modgrid

    
#==================================================
# MCMC: Defines log prior
#==================================================

def lnprior(params, par_min, par_max):
    '''
    Return the flat prior on parameters

    Parameters
    ----------
    - params (list): the parameters
    - par_min (list): the minimum value for params
    - par_max (list): the maximum value for params

    Output
    ------
    - prior (float): the value of the prior, either 0 or -inf

    '''

    prior = 0.0
    
    for i in range(len(params)):
        if params[i] <= par_min[i] or params[i] >= par_max[i] :
            prior = -np.inf
            
    return prior


#==================================================
# MCMC: Defines log likelihood
#==================================================

def lnlike(params, data, modgrid, par_min, par_max, gauss=True):
    '''
    Return the log likelihood for the given parameters

    Parameters
    ----------
    - params (list): the parameters
    - data (Table): the data flux and errors
    - modgrid (Table): grid of model for different scaling to be interpolated
    - par_min (list): the minimum value for params
    - par_max (list): the maximum value for params
    - gauss (bool): use a gaussian approximation for errors

    Output
    ------
    - lnlike (float): the value of the log likelihood
    '''

    #---------- Get the prior
    prior = lnprior(params, par_min, par_max)
    if prior == -np.inf: # Should not go for the model if prior out
        return -np.inf
    if np.isinf(prior):
        return -np.inf
    if np.isnan(prior):
        return -np.inf
    
    #---------- Get the test model
    if params[0] <= 0: # should never happen, but it does, so debug when so
        import pdb
        pdb.set_trace()
        
    test_model = model_specimg(modgrid, params)
    
    #---------- Compute the Gaussian likelihood
    # Gaussian likelihood
    if gauss:
        sigma = np.sqrt(test_model['cluster']+test_model['background'])
        chi2 = (data - test_model['cluster']-test_model['background'])**2/sigma**2
        lnL = -0.5*np.nansum(chi2)

    # Poisson with Bkg
    else:        
        L_i1 = test_model['cluster']+test_model['background']
        L_i2 = data*np.log(test_model['cluster']+test_model['background'])
        lnL  = -np.nansum(L_i1 - L_i2)
        
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

    # Interpolate for flatten grid of parameters
    outf_cl = interpn((modgrid['spa_val'], modgrid['spe_val'],
                       modgrid['e_val'], modgrid['y_val'], modgrid['x_val']),
                      modgrid['models_cl'],
                      (params[1], params[2], modgrid['eef_val'], modgrid['yyf_val'], modgrid['xxf_val']))

    outf_bk = interpn((modgrid['spa_val'], modgrid['spe_val'],
                       modgrid['e_val'], modgrid['y_val'], modgrid['x_val']),
                      modgrid['models_bk'],
                      (params[1], params[2], modgrid['eef_val'], modgrid['yyf_val'], modgrid['xxf_val']))

    # Reshape according to xx
    out_cl = np.reshape(outf_cl, modgrid['xx_val'].shape)
    out_bk = np.reshape(outf_bk, modgrid['xx_val'].shape)
    
    # Add normalization parameter and save
    output_model = {'cluster':params[0]*out_cl, 'background':out_bk}
    
    return output_model


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
                   GaussLike=False,
                   reset_mcmc=False,
                   run_mcmc=True,
                   FWHM=0.1*u.deg,
                   theta=1.0*u.deg,
                   coord=None,
                   profile_reso=0.05*u.deg):
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
    - Nmc (int): number of monte carlo point when resampling the chains
    - GaussLike (bool): use gaussian approximation of the likelihood
    - reset_mcmc (bool): reset the existing MCMC chains?
    - run_mcmc (bool): run the MCMC sampling?  
    - FWHM (quantity): size of the FWHM to be used for smoothing
    - theta (quantity): containment angle for plots
    - coord (SkyCoord): source coordinates for extraction
    - profile_reso (quantity): bin size for profile

    Output
    ------
    The final MCMC chains and plots are saved
    """

    #========== Reset matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    #========== Read the data
    data, modgrid = read_data(input_files)
    
    #========== Guess parameter definition
    # Normalization, scaling profile \propto profile_input^eta, CRp index
    parname = ['X_{CRp}/X_{CRp,0}', '\\eta_{CRp}', '\\alpha_{CRp}'] 
    par0 = np.array([1.0, np.mean(modgrid['spa_val']), np.mean(modgrid['spe_val'])])
    par_min = [0,      np.amin(modgrid['spa_val']), np.amin(modgrid['spe_val'])]
    par_max = [np.inf, np.amax(modgrid['spa_val']), np.amax(modgrid['spe_val'])]

    #========== Names
    sampler_file   = subdir+'/MCMC_sampler.pkl'
    chainstat_file = subdir+'/MCMC_chainstat.txt'
    chainplot_file = subdir+'/MCMC_chainplot'

    #========== Start running MCMC definition and sampling    
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(sampler_file)
    if sampler_exist:
        sampler = mcmc_common.load_object(sampler_file)
        print('    Existing sampler: '+sampler_file)
    
    #---------- MCMC parameters
    ndim = len(par0)
    
    print('--- MCMC profile parameters: ')
    print('    ndim                = '+str(ndim))
    print('    nwalkers            = '+str(nwalkers))
    print('    nsteps              = '+str(nsteps))
    print('    burnin              = '+str(burnin))
    print('    conf                = '+str(conf))
    print('    reset_mcmc          = '+str(reset_mcmc))
    print('    Gaussian likelihood = '+str(GaussLike))

    #---------- Defines the start
    if sampler_exist:
        if reset_mcmc:
            print('--- Reset MCMC even though sampler already exists')
            pos = [par0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
            sampler.reset()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                            args=[data, modgrid, par_min, par_max, GaussLike])
        else:
            print('--- Start from already existing sampler')
            pos = sampler.chain[:,-1,:]
    else:
        print('--- No pre-existing sampler, start from scratch')
        pos = [par0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                        args=[data, modgrid, par_min, par_max, GaussLike])
        
    #---------- Run the MCMC
    if run_mcmc:
        print('--- Runing '+str(nsteps)+' MCMC steps')
        sampler.run_mcmc(pos, nsteps)

    #---------- Save the MCMC after the run
    mcmc_common.save_object(sampler, sampler_file)

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
    
    modelplot(data, Best_model, MC_model, modgrid['header'], modgrid['Ebins'], subdir,
              conf=conf, FWHM=FWHM, theta=theta, coord=coord, profile_reso=profile_reso)
    
