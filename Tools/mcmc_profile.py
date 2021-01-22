"""
This file contains the modules which perform the MCMC constraint. 
"""

#==================================================
# Requested imports
#==================================================

import pickle
import copy
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import seaborn as sns
import pandas as pd
import emcee
import corner
import ctools
import gammalib

from minot.model_tools import trapz_loglog
from minot.ClusterTools.map_tools import radial_profile_cts
from kesacco.Tools import plotting
from kesacco.Tools import mcmc_common
from kesacco.Tools import make_cluster_template
from kesacco.Tools import cubemaking


#==================================================
# Compute the model profile grid
#==================================================

def build_model_grid(cpipe,
                     subdir,
                     rad, prof_ini,
                     spatial_value, spatial_idx,
                     profile_reso,
                     includeIC=False,
                     rm_tmp=False):
    """
    Build a grid of models for the cluster and background, using
    as n_CRp(r) propto n_CRp_ref^scaling.
    
    Parameters
    ----------

    Outputs files
    -------------
    - 

    """
            
    # Save the cluster model before modification
    cluster_tmp = copy.deepcopy(cpipe.cluster)
    
    #----- Loop changing profile
    spatial_npt = len(spatial_value)
    
    for imod in range(spatial_npt):
        print('--- Building model template '+str(1+imod)+'/'+str(spatial_npt))

        #---------- Indexing
        spatial_i = spatial_value[imod]            
        exti = 'TMP_'+str(imod)

        #---------- Compute the model spectrum, map, and xml model file
        # Re-scaling        
        cluster_tmp.density_crp_model  = {'name':'User',
                                          'radius':rad, 'profile':prof_ini.value ** spatial_i}
        
        # Cluster model
        make_cluster_template.make_map(cluster_tmp,
                                       subdir+'/Model_Map_'+exti+'.fits',
                                       Egmin=cpipe.obs_setup.get_emin(),Egmax=cpipe.obs_setup.get_emax(),
                                       includeIC=includeIC)
        make_cluster_template.make_spectrum(cluster_tmp,
                                            subdir+'/Model_Spectrum_'+exti+'.txt',
                                            energy=np.logspace(-1,5,1000)*u.GeV,
                                            includeIC=includeIC)

        # xml model
        model_tot = gammalib.GModels(cpipe.output_dir+'/Ana_Model_Input_Stack.xml')
        clencounter = 0
        for i in range(len(model_tot)):
            if model_tot[i].name() == cluster_tmp.name:
                spefn = subdir+'/Model_Spectrum_'+exti+'.txt'
                model_tot[i].spectral().filename(spefn)
                spafn = subdir+'/Model_Map_'+exti+'.fits'
                model_tot[i].spatial(gammalib.GModelSpatialDiffuseMap(spafn))
                clencounter += 1
        if clencounter != 1:
            raise ValueError('No cluster encountered in the input stack model')
        model_tot.save(subdir+'/Model_Input_'+exti+'.xml')

        #---------- Likelihood fit
        like = ctools.ctlike()
        like['inobs']           = cpipe.output_dir+'/Ana_Countscube.fits'
        like['inmodel']         = subdir+'/Model_Input_'+exti+'.xml'
        like['expcube']         = cpipe.output_dir+'/Ana_Expcube.fits'
        like['psfcube']         = cpipe.output_dir+'/Ana_Psfcube.fits'
        like['bkgcube']         = cpipe.output_dir+'/Ana_Bkgcube.fits'
        like['edispcube']       = cpipe.output_dir+'/Ana_Edispcube.fits'
        like['edisp']           = cpipe.spec_edisp
        like['outmodel']        = subdir+'/Model_Output_'+exti+'.xml'
        like['outcovmat']       = 'NONE'
        like['statistic']       = cpipe.method_stat
        like['refit']           = False
        like['like_accuracy']   = 0.005
        like['max_iter']        = 50
        like['fix_spat_for_ts'] = False
        like['logfile']         = subdir+'/Model_Output_log_'+exti+'.txt'
        like.logFileOpen()
        like.execute()
        like.logFileClose()

        #---------- Compute the 3D residual cube
        cpipe._rm_source_xml(subdir+'/Model_Output_'+exti+'.xml',
                             subdir+'/Model_Output_Cluster_'+exti+'.xml',
                             cluster_tmp.name)
        
        modcube = cubemaking.model_cube(cpipe.output_dir,
                                        cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                        cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                        cpipe.spec_ebinalg,
                                        edisp=cpipe.spec_edisp,
                                        stack=cpipe.method_stack,
                                        silent=True,
                                        logfile=subdir+'/Model_Cube_log_'+exti+'.txt',
                                        inmodel_usr=subdir+'/Model_Output_'+exti+'.xml',
                                        outmap_usr=subdir+'/Model_Cube_'+exti+'.fits')
        
        modcube_Cl = cubemaking.model_cube(cpipe.output_dir,
                                           cpipe.map_reso, cpipe.map_coord, cpipe.map_fov,
                                           cpipe.spec_emin, cpipe.spec_emax, cpipe.spec_enumbins,
                                           cpipe.spec_ebinalg,
                                           edisp=cpipe.spec_edisp,
                                           stack=cpipe.method_stack, silent=True,
                                           logfile=subdir+'/Model_Cube_Cluster_log_'+exti+'.txt',
                                           inmodel_usr=subdir+'/Model_Output_Cluster_'+exti+'.xml',
                                           outmap_usr=subdir+'/Model_Cube_Cluster_'+exti+'.fits')
    
    #----- Build the data
    hdul = fits.open(cpipe.output_dir+'/Ana_Countscube.fits')
    header = hdul[0].header
    header.remove('NAXIS3')
    header['NAXIS'] = 2
    cntmap = np.sum(hdul[0].data, axis=0)
    hdul.close()
    r_dat, p_dat, err_dat = radial_profile_cts(cntmap,
                                               [cpipe.cluster.coord.icrs.ra.to_value('deg'),
                                                cpipe.cluster.coord.icrs.dec.to_value('deg')],
                                               stddev=np.sqrt(cntmap), header=header,
                                               binsize=profile_reso.to_value('deg'),
                                               stat='POISSON', counts2brightness=True)
    tabdat = Table()
    tabdat['radius']      = r_dat
    tabdat['radius_min']  = r_dat - profile_reso.to_value('deg')/2.0
    tabdat['radius_max']  = r_dat + profile_reso.to_value('deg')/2.0
    tabdat['profile']     = p_dat
    
    dat_hdu = fits.BinTableHDU(tabdat)
    
    #----- Build the grid
    modgrid_bk = np.zeros((spatial_npt, len(p_dat)))
    modgrid_cl = np.zeros((spatial_npt, len(p_dat)))
    
    for imod in range(spatial_npt):
        exti = 'TMP_'+str(imod)
        
        # Extract the profile
        hdul = fits.open(subdir+'/Model_Cube_'+exti+'.fits')
        cntmap = np.sum(hdul[0].data, axis=0)
        hdul.close()
        
        hdul_cl = fits.open(subdir+'/Model_Cube_Cluster_'+exti+'.fits')
        cntmap_cl = np.sum(hdul_cl[0].data, axis=0)
        hdul_cl.close()
        
        map_cl = cntmap - cntmap_cl
        map_bk = cntmap_cl
        
        r_cl, p_cl, err_cl = radial_profile_cts(map_cl,
                                                [cpipe.cluster.coord.icrs.ra.to_value('deg'),
                                                 cpipe.cluster.coord.icrs.dec.to_value('deg')],
                                                stddev=np.sqrt(map_cl), header=header,
                                                binsize=profile_reso.to_value('deg'),
                                                stat='POISSON', counts2brightness=True)
        r_bk, p_bk, err_bk = radial_profile_cts(map_bk,
                                                [cpipe.cluster.coord.icrs.ra.to_value('deg'),
                                                 cpipe.cluster.coord.icrs.dec.to_value('deg')],
                                                stddev=np.sqrt(map_bk), header=header,
                                                binsize=profile_reso.to_value('deg'),
                                                stat='POISSON', counts2brightness=True)
        modgrid_bk[imod,:] = p_bk
        modgrid_cl[imod,:] = p_cl

    grid_cl_hdu = fits.ImageHDU(modgrid_cl)
    grid_bk_hdu = fits.ImageHDU(modgrid_bk)
    
    #----- Make the index table
    scal = Table()
    scal['index'] = spatial_idx
    scal['value'] = spatial_value
    scal_hdu = fits.BinTableHDU(scal)
    
    #----- Make and save HDUlist
    hdul = fits.HDUList()
    hdul.append(scal_hdu)
    hdul.append(dat_hdu)
    hdul.append(grid_cl_hdu)
    hdul.append(grid_bk_hdu)
    hdul.writeto(subdir+'/Grid_Sampling.fits', overwrite=True)

    #----- Save the properties of the last computation run
    np.save(subdir+'/Grid_Parameters.npy',
            [cpipe.cluster, spatial_value], allow_pickle=True)
    
    #----- remove TMP files
    if rm_tmp:
        for imod in range(spatial_npt):
            exti = 'TMP_'+str(imod)
            os.remove(subdir+'/Model_Map_'+exti+'.fits')
            os.remove(subdir+'/Model_Spectrum_'+exti+'.txt')
            os.remove(subdir+'/Model_Cube_'+exti+'.fits')
            os.remove(subdir+'/Model_Cube_log_'+exti+'.txt')
            os.remove(subdir+'/Model_Cube_Cluster_'+exti+'.fits')
            os.remove(subdir+'/Model_Cube_Cluster_log_'+exti+'.txt')
            os.remove(subdir+'/Model_Input_'+exti+'.xml')
            os.remove(subdir+'/Model_Output_'+exti+'.xml')
            os.remove(subdir+'/Model_Output_log_'+exti+'.txt')
            os.remove(subdir+'/Model_Output_Cluster_'+exti+'.xml')


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
    MC_model (ndarray): Nmc x N_eng array

    """

    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1],
                                    param_chains.shape[2])
    
    Nsample = len(par_flat[:,0])-1
    
    MC_model_background = np.zeros((Nmc, len(modgrid['radius'])))
    MC_model_cluster = np.zeros((Nmc, len(modgrid['radius'])))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        models = model_profile(modgrid, param_MC)
        MC_model_cluster[i,:]    = models['cluster']
        MC_model_background[i,:] = models['background']

    MC_models = {'cluster':MC_model_cluster,
                 'background':MC_model_background}
    
    return MC_models

    
#==================================================
# Plot the output fit model
#==================================================

def modelplot(data, Best_model, MC_model, subdir, conf=68.0):
    """
    Plot the data versus model and constraints
        
    Parameters
    ----------

    Output
    ------
    Plots are saved
    """
    
    #========== Extract relevant info
    Nmc = MC_model['cluster'].shape[0]
    radius = data['radius']
    area = np.pi*(data['radius_max']**2-data['radius_min']**2)
    MC_perc_cl  = np.percentile(MC_model['cluster'],
                                [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    MC_perc_bk  = np.percentile(MC_model['background'],
                                [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    MC_perc_tot = np.percentile(MC_model['background']+MC_model['cluster'],
                                [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    
    #========== Plot
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(data['radius'])/2.0, np.amax(data['radius'])*2.0]
    rngyp = 1.2*np.nanmax(data['profile']-Best_model['background'])
    rngym = 0.5*np.nanmin((data['profile']-Best_model['background'])[data['profile']-Best_model['background'] > 0])
    ylim = [rngym, rngyp]

    ax1.plot(radius, Best_model['cluster'],     ls='-', linewidth=2, color='k', label='Maximum likelihood model')
    ax1.plot(radius, MC_perc_cl[1,:], ls='--', linewidth=2, color='b', label='Median')
    ax1.plot(radius, MC_perc_cl[0,:], ls=':', linewidth=1, color='b')
    ax1.plot(radius, MC_perc_cl[2,:], ls=':', linewidth=1, color='b')
    ax1.fill_between(radius, MC_perc_cl[0,:], y2=MC_perc_cl[2,:], alpha=0.2, color='blue', label=str(conf)+'% CL')
    for i in range(Nmc):
        ax1.plot(radius, MC_model['cluster'][i,:], ls='-', linewidth=0.5, alpha=0.1, color='blue')

    ax1.errorbar(data['radius'], data['profile']-Best_model['background'], yerr=(area*data['profile'])**0.5/area,
                 marker='o', elinewidth=2, color='red',
                 markeredgecolor="black", markerfacecolor="red",
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
    ax2.plot(radius, (Best_model['cluster']*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
    ax2.set_ylabel('Profile (arcmin$^{-2}$)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                 (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
    
    # Residual plot
    resi = data['profile']-Best_model['background']-Best_model['cluster']
    ax3.plot(data['radius'], resi/((area*data['profile'])**0.5/area),
             linestyle='', marker='o', color='k')
    ax3.plot(radius,  radius*0, linestyle='-', color='k')
    ax3.plot(radius,  radius*0+2, linestyle='--', color='k')
    ax3.plot(radius,  radius*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('Radius (deg)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(subdir+'/MCMC_fit_results_bkgsub.pdf')
    plt.close()

    #========== Plot
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(data['radius'])/2.0, np.amax(data['radius'])*2.0]
    rngyp = 1.2*np.nanmax(data['profile'])
    rngym = 0.5*np.nanmin((data['profile'])[data['profile'] > 0])
    ylim = [rngym, rngyp]

    ax1.plot(radius, Best_model['cluster']+Best_model['background'],
             ls='-', linewidth=2, color='k', label='Maximum likelihood model')
    ax1.plot(radius, MC_perc_cl[1,:], ls='--', linewidth=2, color='b', label='Median (cluster)')
    ax1.plot(radius, MC_perc_cl[0,:], ls=':', linewidth=1, color='b')
    ax1.plot(radius, MC_perc_cl[2,:], ls=':', linewidth=1, color='b')
    ax1.fill_between(radius, MC_perc_cl[0,:], y2=MC_perc_cl[2,:], alpha=0.2, color='blue')

    ax1.plot(radius, MC_perc_bk[1,:], ls='--', linewidth=2, color='g', label='Median (background)')
    ax1.plot(radius, MC_perc_bk[0,:], ls=':', linewidth=1, color='g')
    ax1.plot(radius, MC_perc_bk[2,:], ls=':', linewidth=1, color='g')
    ax1.fill_between(radius, MC_perc_bk[0,:], y2=MC_perc_bk[2,:], alpha=0.2, color='green')

    ax1.plot(radius, MC_perc_tot[1,:], ls='--', linewidth=2, color='grey', label='Median (Total)')
    ax1.plot(radius, MC_perc_tot[0,:], ls=':', linewidth=1, color='grey')
    ax1.plot(radius, MC_perc_tot[2,:], ls=':', linewidth=1, color='grey')
    ax1.fill_between(radius, MC_perc_tot[0,:], y2=MC_perc_tot[2,:],
                     alpha=0.2, color='grey', label=str(conf)+'% CL')
    for i in range(Nmc):
        ax1.plot(radius, MC_model['cluster'][i,:], ls='-', linewidth=0.5, alpha=0.1, color='blue')

    for i in range(Nmc):
        ax1.plot(radius, MC_model['background'][i,:], ls='-', linewidth=0.5, alpha=0.1, color='green')

    for i in range(Nmc):
        ax1.plot(radius, MC_model['background'][i,:]+MC_model['cluster'][i,:],
                 ls='-', linewidth=0.5, alpha=0.1, color='grey')
        
    ax1.errorbar(data['radius'], data['profile'], yerr=(area*data['profile'])**0.5/area,
                 marker='o', elinewidth=2, color='red',
                 markeredgecolor="black", markerfacecolor="red",
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
    ax2.plot(radius, ((Best_model['cluster']+Best_model['background'])*u.Unit('deg-2')).to_value('arcmin-2'),
             'k-', alpha=0.0)
    ax2.set_ylabel('Profile (arcmin$^{-2}$)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                 (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
    
    # Residual plot
    resi = data['profile']-Best_model['background']-Best_model['cluster']
    ax3.plot(data['radius'], resi/((area*data['profile'])**0.5/area),
             linestyle='', marker='o', color='k')
    ax3.plot(radius,  radius*0, linestyle='-', color='k')
    ax3.plot(radius,  radius*0+2, linestyle='--', color='k')
    ax3.plot(radius,  radius*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('Radius (deg)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(subdir+'/MCMC_fit_results_bkginc.pdf')
    plt.close()


#==================================================
# Read the data
#==================================================

def read_data(input_file):
    """
    Read the data to extract the necessary information
    
    Parameters
    ----------
    - specfile (str): file where the data is stored

    Output
    ------
    - data (Table): Table containing the data

    """

    hdul = fits.open(input_file)

    modgrid = {'spa_val':hdul[1].data['value'],
               'radius':hdul[2].data['radius'],
               'radius_min':hdul[2].data['radius_min'],
               'radius_max':hdul[2].data['radius_max'],
               'models_cl':hdul[3].data,
               'models_bk':hdul[4].data}

    data = {'radius':hdul[2].data['radius'],
            'radius_min':hdul[2].data['radius_min'],
            'radius_max':hdul[2].data['radius_max'],
            'profile':hdul[2].data['profile']}

    hdul.close()
 
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
        
    test_model = model_profile(modgrid, params)
    
    #---------- Compute the Gaussian likelihood
    # Gaussian likelihood
    if gauss:
        resi = data['profile']-test_model['cluster']-test_model['background']
        chi2 = resi**2/np.sqrt(test_model['cluster'])**2
        lnL = -0.5*np.nansum(chi2)

    # Likelihood taking into account the background counts
    else:
        area = np.pi*data['radius_max']**2 - np.pi*data['radius_min']**2
        L_i1 = (test_model['cluster']+test_model['background'])*area
        L_i2 = data['profile']*area * np.log(L_i1)
        lnL  = -np.nansum(L_i1 - L_i2)
        
    # In case of NaN, goes to infinity
    if np.isnan(lnL):
        lnL = -np.inf
        
    return lnL + prior


#==================================================
# MCMC: Defines model
#==================================================

def model_profile(modgrid, params):
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
    
    f_cl = interp1d(modgrid['spa_val'], modgrid['models_cl'], axis=0)
    f_bk = interp1d(modgrid['spa_val'], modgrid['models_bk'], axis=0)

    output_model_cl = params[0] * f_cl(params[1])
    output_model_bk = f_bk(params[1])

    output_model = {'cluster':output_model_cl, 'background':output_model_bk}
    
    return output_model


#==================================================
# MCMC: run the fit
#==================================================

def run_profile_constraint(input_file,
                           subdir,
                           nwalkers=10,
                           nsteps=1000,
                           burnin=100,
                           conf=68.0,
                           Nmc=100,
                           GaussLike=False,
                           reset_mcmc=False,
                           run_mcmc=True):
    """
    Run the MCMC constraints to the profile
        
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

    Output
    ------
    The final MCMC chains and plots are saved
    """

    #========== Reset matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    #========== Read the data
    data, modgrid = read_data(input_file)
    
    #========== Guess parameter definition
    parname = ['X_{CRp}/X_{CRp,0}', '\\eta_{CRp}'] # Normalization and scaling prof \propto prof_input^eta
    par0 = np.array([1.0, np.mean(modgrid['spa_val'])])
    par_min = [0,      np.amin(modgrid['spa_val'])]
    par_max = [np.inf, np.amax(modgrid['spa_val'])]

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
    Best_model = model_profile(modgrid, par_best)

    #---------- Plots and results
    mcmc_common.chains_plots(param_chains, parname, chainplot_file,
                             par_best=par_best, par_percentile=par_percentile, conf=conf,
                             par_min=par_min, par_max=par_max)
    
    modelplot(data, Best_model, MC_model, subdir, conf=conf)
