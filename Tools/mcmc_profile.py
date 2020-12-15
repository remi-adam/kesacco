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

from minot.model_tools import trapz_loglog
from kesacco.Tools import plotting


#==================================================
# Save object
#==================================================

def save_object(obj, filename):
    '''
    Save MCMC object

    Parameters
    ----------
    - obj (object): python object, in this case a MCMC emcee object
    - filename (str): file where to save the object

    Output
    ------
    - Object saved as filename
    '''
    
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        
#==================================================
# Restore object
#==================================================

def load_object(filename):
    '''
    Restore MCMC object

    Parameters
    ----------
    - filename (str): file to restore

    Output
    ------
    - obj: Object saved in filename
    '''
    
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        
    return obj


#==================================================
# Compute chain statistics
#==================================================

def chains_statistics(param_chains, lnL_chains, parname=None, conf=68.0, show=True,
                      outfile=None):
    """
    Get the statistics of the chains, such as maximum likelihood,
    parameters errors, etc.
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - lnl_chains (np array): log likelihood values corresponding to the chains
    - parname (list): list of parameter names
    - conf (float): confidence interval in %
    - show (bool): show or not the values
    - outfile (str): full path to file to write results

    Output
    ------
    - par_best (float): best-fit parameter
    - par_percentile (list of float): median, lower bound at CL, upper bound at CL
    
    """
    
    if outfile is not None:
        file = open(outfile,'w')
    
    Npar = len(param_chains[0,0,:])

    wbest = (lnL_chains == np.amax(lnL_chains))
    par_best       = np.zeros(Npar)
    par_percentile = np.zeros((3, Npar))
    for ipar in range(Npar):
        # Maximum likelihood
        par_best[ipar]          = param_chains[:,:,ipar][wbest][0]

        # Median and xx % CL
        perc = np.percentile(param_chains[:,:,ipar].flatten(),
                             [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])
        par_percentile[:, ipar] = perc
        if show:
            if parname is not None:
                parnamei = parname[ipar]
            else:
                parnamei = 'no name'

            q = np.diff(perc)
            txt = "{0}_{{-{1}}}^{{{2}}}"
            txt = txt.format(perc[1], q[0], q[1])
            
            print('param '+str(ipar)+' ('+parnamei+'): ')
            print('   median   = '+str(perc[1])+' -'+str(perc[1]-perc[0])+' +'+str(perc[2]-perc[1]))
            print('   best-fit = '+str(par_best[ipar])+' -'+str(par_best[ipar]-perc[0])+' +'+str(perc[2]-par_best[ipar]))
            print('   '+parnamei+' = '+txt)

            if outfile is not None:
                file.write('param '+str(ipar)+' ('+parnamei+'): '+'\n')
                file.write('   median   = '+str(perc[1])+' -'+str(perc[1]-perc[0])+' +'+str(perc[2]-perc[1])+'\n')
                file.write('   best-fit = '+str(par_best[ipar])+' -'+str(par_best[ipar]-perc[0])+' +'+str(perc[2]-par_best[ipar])+'\n')
                file.write('   '+parnamei+' = '+txt+'\n')

    if outfile is not None:
        file.close() 
            
    return par_best, par_percentile


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
    
    MC_model = np.zeros((Nmc, len(modgrid['models'][0,:])))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        MC_model[i,:] = model_profile(modgrid, param_MC)
    
    return MC_model

#==================================================
# Plots related to the chains
#==================================================

def chainplots(param_chains, parname, rout_file,
               par_best=None, par_percentile=None, conf=68.0,
               par_min=None, par_max=None):
    """
    Plot related to chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - rout_file (str): rout file  where to save plots
    - par_best (float): best-fit parameter
    - par_percentile (list of float): median, lower bound at CL, upper bound at CL
    - conf (float): confidence interval in %

    Output
    ------
    Plots are saved in the output directory

    """

    Nbin_hist = 40
    Npar = len(param_chains[0,0,:])
    Nchain = len(param_chains[:,0,0])

    # Chain histogram
    for ipar in range(Npar):
        if par_best is not None:
            par_besti = par_best[ipar]
        plotting.seaborn_1d(param_chains[:,:,ipar].flatten(),
                            output_fig=rout_file+'_MCMC_chain_histo'+str(ipar)+'.pdf',
                            ci=0.68, truth=None, best=par_besti,
                            label='$'+parname[ipar]+'$',
                            gridsize=100, alpha=(0.2, 0.4), 
                            figsize=(10,10), fontsize=12,
                            cols=[('blue','grey', 'orange')])
        plt.close("all")

    # Chains
    fig, axes = plt.subplots(Npar, figsize=(8, 2*Npar), sharex=True)
    for i in range(Npar):
        ax = axes[i]
        for j in range(Nchain):
            ax.plot(param_chains[j, :, i], alpha=0.5)
        ax.set_xlim(0, len(param_chains[0,:,0]))
        ax.set_ylabel('$'+parname[i]+'$')
    axes[-1].set_xlabel("step number")
    fig.savefig(rout_file+'_MCMC_chains.pdf')
    plt.close()

    # Corner plot using seaborn
    parname_corner = []
    for i in range(Npar): parname_corner.append('$'+parname[i]+'$')
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
    df = pd.DataFrame(par_flat, columns=parname_corner)
    plotting.seaborn_corner(df, output_fig=rout_file+'_MCMC_triangle_seaborn.pdf',
                            n_levels=30, cols=[('royalblue', 'k', 'grey', 'Blues')], 
                            ci2d=[0.68, 0.95], gridsize=100,
                            linewidth=2.0, alpha=(0.1, 0.3, 1.0), figsize=((Npar+1)*3,(Npar+1)*3))
    plt.close("all")
    
    # Corner plot using corner
    figure = corner.corner(par_flat,
                           bins=Nbin_hist,
                           color='k',
                           smooth=1,
                           labels=parname_corner,
                           quantiles=(0.16, 0.84))
    figure.savefig(rout_file+'_MCMC_triangle_corner.pdf')
    plt.close("all")

    
#==================================================
# Plot the output fit model
#==================================================

def modelplot(cluster_test, data, modgrid, par_best, param_chains, MC_model, conf=68.0, Nmc=100):
    """
    Plot the data versus model and constraints
        
    Parameters
    ----------

    Output
    ------
    Plots are saved
    """
    
    #========== Extract relevant info    
    bf_model = model_profile(modgrid, par_best)
    MC_perc  = np.percentile(MC_model, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    radius = data['radius']
    
    #========== Plot
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(data['radius'])/2.0, np.amax(data['radius'])*2.0]
    rngyp = 1.2*np.nanmax(data['profile']+data['error'])
    rngym = 0.5*np.nanmin((data['profile'])[data['profile'] > 0])
    ylim = [rngym, rngyp]

    ax1.plot(radius, bf_model,     ls='-', linewidth=2, color='k', label='Maximum likelihood model')
    ax1.plot(radius, MC_perc[1,:], ls='--', linewidth=2, color='b', label='Median')
    ax1.plot(radius, MC_perc[0,:], ls=':', linewidth=1, color='b')
    ax1.plot(radius, MC_perc[2,:], ls=':', linewidth=1, color='b')
    ax1.fill_between(radius, MC_perc[0,:], y2=MC_perc[2,:], alpha=0.2, color='blue', label=str(conf)+'% CL')
    for i in range(Nmc):
        ax1.plot(radius, MC_model[i,:], ls='-', linewidth=0.5, alpha=0.1, color='blue')

    ax1.errorbar(data['radius'], data['profile'], yerr=data['error'],
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
    ax2.plot(radius, (bf_model*u.Unit('deg-2')).to_value('arcmin-2'), 'k-', alpha=0.0)
    ax2.set_ylabel('Profile (arcmin$^{-2}$)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('deg-2')).to_value('arcmin-2'),
                 (ylim[1]*u.Unit('deg-2')).to_value('arcmin-2'))
    
    # Residual plot
    ax3.plot(data['radius'], (data['profile']-bf_model)/data['error'],
             linestyle='', marker='o', color='k')
    ax3.plot(radius,  radius*0, linestyle='-', color='k')
    ax3.plot(radius,  radius*0+2, linestyle='--', color='k')
    ax3.plot(radius,  radius*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-3, 3)
    ax3.set_xlabel('Radius (deg)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_results.pdf')
    plt.close()


#==================================================
# Read the data
#==================================================

def read_data(prof_files):
    """
    Read the data to extract the necessary information
    
    Parameters
    ----------
    - specfile (str): file where the data is stored

    Output
    ------
    - data (Table): Table containing the data

    """

    # Get measured data
    hdu = fits.open(prof_files[0])
    measured = hdu[1].data
    hdu.close()

    # Get expected
    hdu = fits.open(prof_files[1])
    sampling = hdu[1].data
    models   = hdu[2].data
    hdu.close()
    
    # Check that the radius is the same, as expected
    if np.sum(measured['radius'] - models['radius']) > 0:
        print('!!!!! WARNING: it is possible that we have a problem with radius !!!!!')
    if len(models['radius'])-len(measured['radius']) != 0:
        print('!!!!! WARNING: it is possible that we have a problem with radius !!!!!')
    
    # Extract quantities and fill data
    data = Table()
    
    radius   = measured['radius']*u.deg
    profile  = measured['profile']*u.deg**-2
    error    = measured['error']*u.deg**-2
    
    data['radius']   = radius.to_value('deg')
    data['profile']  = profile.to_value('deg-2')
    data['error']    = error.to_value('deg-2')

    # Extract and fill model_grid
    model_list = []
    for i in range(len(sampling['index'])):
        model_list.append(models['profile'+str(i)])
    
    modgrid = {'sampling':sampling,            # information about the scaling used for the model
               'radius':models['radius'],      # radius correspondign for the models
               'models':np.array(model_list)}  # models grid (vs radius and scaling)
    
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

def lnlike(params, data, modgrid, par_min, par_max):
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
    chi2 = (data['profile'] - test_model)**2 / data['error']**2
    lnL = -0.5*np.nansum(chi2)
    
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
    
    f = interp1d(modgrid['sampling']['value'], modgrid['models'].T, axis=1)

    output_model = params[0] * f(params[1])
    
    return output_model


#==================================================
# MCMC: run the fit
#==================================================

def run_profile_constraint(cluster_test,
                           profile_file,
                           nwalkers=10,
                           nsteps=1000,
                           burnin=100,
                           conf=68.0,
                           Nmc=100,
                           reset_mcmc=False,
                           run_mcmc=True):
    """
    Run the MCMC constraints to the profile
        
    Parameters
    ----------
    - cluster_test (minot object): a cluster to be used when sampling parameters of model
    - profile_file (str): full path to the profile data and expected model
    - nwalkers (int): number of emcee wlakers
    - nsteps (int): number of emcee MCMC steps
    - burnin (int): number of point to remove assuming it is burnin
    - conf (float): confidence limit percentage for results
    - Nmc (int): number of monte carlo point when resampling the chains
    - reset_mcmc (bool): reset the existing MCMC chains?
    - run_mcmc (bool): run the MCMC sampling?                            
    - Emin/Emax (flaot, GeV): Energy min and max for flux/luminosity computation

    Output
    ------
    The final MCMC chains and plots are saved
    """

    #========== Reset matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    #========== Read the data
    data, modgrid = read_data(profile_file)
    
    #========== Guess parameter definition
    par0 = np.array([1.0, 1.0])
    parname = ['X_{CRp}/X_{CRp, input}', '\\eta_{CRp}'] # Normalization and scaling profile \propto profile_input^eta
    par_min = [0,      np.amin(modgrid['sampling']['value'])]
    par_max = [np.inf, np.amax(modgrid['sampling']['value'])]
    
    #========== Start running MCMC definition and sampling    
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_sampler.pkl')
    if sampler_exist:
        sampler = load_object(cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_sampler.pkl')
        print('    Existing sampler: '+cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_sampler.pkl')
    
    #---------- MCMC parameters
    ndim = len(par0)
    
    print('--- MCMC profile parameters: ')
    print('    ndim       = '+str(ndim))
    print('    nwalkers   = '+str(nwalkers))
    print('    nsteps     = '+str(nsteps))
    print('    burnin     = '+str(burnin))
    print('    conf       = '+str(conf))
    print('    reset_mcmc = '+str(reset_mcmc))

    #---------- Defines the start
    if sampler_exist:
        if reset_mcmc:
            print('--- Reset MCMC even though sampler already exists')
            pos = [par0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
            #pos = [np.random.uniform(par_min[i],par_max[i], nwalkers) for i in range(ndim)]
            #pos = list(np.array(pos).T.reshape((nwalkers,ndim)))
            sampler.reset()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                            args=[data, modgrid, par_min, par_max])
        else:
            print('--- Start from already existing sampler')
            pos = sampler.chain[:,-1,:]
    else:
        print('--- No pre-existing sampler, start from scratch')
        pos = [par0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
        #pos = [np.random.uniform(par_min[i],par_max[i], nwalkers) for i in range(ndim)]
        #pos = list(np.array(pos).T.reshape((nwalkers,ndim)))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                        args=[data, modgrid, par_min, par_max])
        
    #---------- Run the MCMC
    if run_mcmc:
        print('--- Runing '+str(nsteps)+' MCMC steps')
        sampler.run_mcmc(pos, nsteps)

    #---------- Save the MCMC after the run
    save_object(sampler, cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_sampler.pkl')

    #---------- Burnin
    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains = sampler.lnprobability[:, burnin:]
    
    #---------- Get the parameter statistics
    par_best, par_percentile = chains_statistics(param_chains, lnL_chains,
                                                 parname=parname, conf=conf, show=True,
                                                 outfile=cluster_test.output_dir+'/Ana_ResmapCluster_profile_MCMC_chainstat.txt')

    #---------- Get the well-sampled models
    MC_model   = get_mc_model(modgrid, param_chains, Nmc=Nmc)
    Best_model = model_profile(modgrid, par_best)

    #---------- Plots and results
    chainplots(param_chains, parname, cluster_test.output_dir+'/Ana_ResmapCluster_profile',
               par_best=par_best, par_percentile=par_percentile, conf=conf,
               par_min=par_min, par_max=par_max)

    modelplot(cluster_test, data, modgrid, par_best, param_chains, MC_model, conf=conf)
