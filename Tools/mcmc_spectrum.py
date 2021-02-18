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
from kesacco.Tools import mcmc_common


#==================================================
# Compute the constraint on L and F
#==================================================

def get_global_prop(MC_eng, MC_model, Best_model, Dlum, rout_file,
                    Emin=None, Emax=None,
                    conf=68.0, outfile=None):
    """
    Compute constraints on Lumi and Flux
        
    Parameters
    ----------
    - MC_eng (array): an array of energy
    - MC_model (ndarray): Nmc x N_eng array

    Output
    ------
    - Print the values obtained

    """

    # Checking energy range
    if Emin is None:
        Emin = np.amin(MC_eng)
    if Emax is None:
        Emax = np.amax(MC_eng)
    if Emin < np.amin(MC_eng):
        Emin = np.amin(MC_eng)
        print('WARNING: Emin cannot be smaller than min(MC_eng), setting Emin = min(MC_eng)')
    if Emax > np.amax(MC_eng):
        Emax = np.amax(MC_eng)
        print('WARNING: Emax cannot be larger than max(MC_eng), setting Emax = max(MC_eng)')
   
    # Define variables
    Nmc = MC_model.shape[0]
    eng_flux = np.logspace(np.log10(Emin), np.log10(Emax), 100)
    
    L1_mc = np.zeros(Nmc)
    L2_mc = np.zeros(Nmc)
    F1_mc = np.zeros(Nmc)
    F2_mc = np.zeros(Nmc)
    
    for i in range(Nmc):
        # Interpolate the model over the requiered range
        itpval = np.log10(MC_model[i,:]/(MC_eng*1e3)**2)
        itpval[MC_model[i,:] <= 0] = -100
        itpl = interp1d(np.log10(MC_eng), itpval, kind='cubic')
        model_i = 10**itpl(np.log10(eng_flux)) # MeV-1 s-1 cm-2
        
        # compute the integral
        F1_i = trapz_loglog(model_i, eng_flux*1e3)          # ph/s/cm2
        F2_i = trapz_loglog((eng_flux*1e3)*model_i, eng_flux*1e3) # MeV/s/cm2
        
        # store the Luminosity and flux
        F1_mc[i] = F1_i
        F2_mc[i] = F2_i
        L1_mc[i] = F1_i * (4*np.pi*Dlum**2) # ph/s
        L2_mc[i] = F2_i * (4*np.pi*Dlum**2) # MeV/s

    MeV2erg = (1.0*u.MeV).to_value('erg')
    F1_perc = np.percentile(F1_mc, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])           # ph/s/cm2
    F2_perc = np.percentile(F2_mc, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])           # MeV/s/cm2
    F3_perc = np.percentile(F2_mc*MeV2erg, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])   # erg/s/cm2
    L1_perc = np.percentile(L1_mc, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])           # ph/s
    L2_perc = np.percentile(L2_mc, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])           # MeV/s
    L3_perc = np.percentile(L2_mc*MeV2erg, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])   # erg/s

    # Get also the best model
    itpval = np.log10(Best_model/(MC_eng*1e3)**2)
    itpval[Best_model <= 0] = -100
    itpl = interp1d(np.log10(MC_eng), itpval, kind='cubic')
    model_b = 10**itpl(np.log10(eng_flux)) # MeV-1 cm-2 s-1
    F1_B = trapz_loglog(model_b, eng_flux*1e3)
    F2_B = trapz_loglog((eng_flux*1e3)*model_b, eng_flux*1e3)
    F3_B = F2_B*MeV2erg
    L1_B = F1_B * (4*np.pi*Dlum**2) # ph/s
    L2_B = F2_B * (4*np.pi*Dlum**2) # MeV/s
    L3_B = L2_B*MeV2erg

    # Provide information
    if outfile is not None:
        file = open(outfile,'w')
    
    txt = "{0}_{{-{1}}}^{{{2}}}"

    print('')
    print('----- Flux and luminosity information (with Nmc = '+str(Nmc)+'):')
    print('Energy range: '+str(Emin)+' - '+str(Emax)+' GeV')
    if outfile is not None:
        file.write('----- Flux and luminosity information (with Nmc = '+str(Nmc)+'):\n')
        file.write('Energy range: '+str(Emin)+' - '+str(Emax)+' GeV\n')

    t1 = txt.format(F1_perc[1], F1_perc[1]-F1_perc[0], F1_perc[2]-F1_perc[1])
    t2 = txt.format(F1_B, F1_B-F1_perc[0], F1_perc[2]-F1_B)
    print('Flux (ph/s/cm2) median   = '+t1)
    print('                best fit = '+t2)
    if outfile is not None:
        file.write('Flux (ph/s/cm2) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')

    t1 = txt.format(F2_perc[1], F2_perc[1]-F2_perc[0], F2_perc[2]-F2_perc[1])
    t2 = txt.format(F2_B, F2_B-F2_perc[0], F2_perc[2]-F2_B)
    print('Flux (MeV/s/cm2) median   = '+t1)
    print('                 best fit = '+t2)
    if outfile is not None:
        file.write('Flux (MeV/s/cm2) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')
        
    t1 = txt.format(F3_perc[1], F3_perc[1]-F3_perc[0], F3_perc[2]-F3_perc[1])
    t2 = txt.format(F3_B, F3_B-F3_perc[0], F3_perc[2]-F3_B)
    print('Flux (erg/s/cm2) median   = '+t1)
    print('                 best fit = '+t2)
    if outfile is not None:
        file.write('Flux (erg/s/cm2) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')
        
    t1 = txt.format(L1_perc[1], L1_perc[1]-L1_perc[0], L1_perc[2]-L1_perc[1])
    t2 = txt.format(L1_B, L1_B-L1_perc[0], L1_perc[2]-L1_B)
    print('Luminosity (ph/s) median   = '+t1)
    print('                  best fit = '+t2)
    if outfile is not None:
        file.write('Luminosity (ph/s) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')
        
    t1 = txt.format(L2_perc[1], L2_perc[1]-L2_perc[0], L2_perc[2]-L2_perc[1])
    t2 = txt.format(L2_B, L2_B-L2_perc[0], L2_perc[2]-L2_B)
    print('Luminosity (MeV/s) median   = '+t1)
    print('                   best fit = '+t2)
    if outfile is not None:
        file.write('Luminosity (MeV/s) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')
        
    t1 = txt.format(L3_perc[1], L3_perc[1]-L3_perc[0], L3_perc[2]-L3_perc[1])
    t2 = txt.format(L3_B, L3_B-L3_perc[0], L3_perc[2]-L3_B)
    print('Luminosity (erg/s) median   = '+t1)
    print('                   best fit = '+t2)
    if outfile is not None:
        file.write('Luminosity (erg/s) \n')
        file.write('      median   = '+t1+'\n')
        file.write('      best fit = '+t2+'\n')

    if outfile is not None:
        file.close()

    # Plot of histogram
    plotting.seaborn_1d(F1_mc, output_fig=rout_file+'_Flux.pdf',
                        ci=0.68, truth=None,
                        best=None, label='Flux (ph s$^{-1}$ cm$^{-2}$)',
                        gridsize=100, alpha=(0.2, 0.4), 
                        figsize=(10,10), fontsize=12,
                        cols=[('blue','grey', 'orange')])
    plt.close("all")

    
#==================================================
# Plot the output fit model
#==================================================

def modelplot(data, cluster_test, par_best, param_chains, MC_eng, MC_model, conf=68.0, Nmc=100):
    """
    Plot the data versus model and constraints
        
    Parameters
    ----------

    Output
    ------
    Plots are saved
    """

    #========== Extract relevant info    
    bf_model1 = model_dNdEdSdt(cluster_test, data['e_ref'], par_best)
    bf_model2 = model_dNdEdSdt(cluster_test, MC_eng, par_best)
    MC_perc   = np.percentile(MC_model, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    
    #========== Plot
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(MC_eng), np.amax(MC_eng)]
    rngyp1 = 1.2*np.nanmax(data['e2dnde']+data['e2dnde_err'])
    rngyp2 = 1.2*np.nanmax(bf_model2)
    rngyp  = np.amax(np.array([rngyp1,rngyp2]))
    rngym = 0.5*np.nanmin(data['e2dnde'])
    ylim = [rngym, rngyp]

    ax1.plot(MC_eng, bf_model2,    ls='-', linewidth=2, color='k', label='Maximum likelihood model')
    ax1.plot(MC_eng, MC_perc[1,:], ls='--', linewidth=2, color='b', label='Median')
    ax1.plot(MC_eng, MC_perc[0,:], ls=':', linewidth=1, color='b')
    ax1.plot(MC_eng, MC_perc[2,:], ls=':', linewidth=1, color='b')
    ax1.fill_between(MC_eng, MC_perc[0,:], y2=MC_perc[2,:], alpha=0.2, color='blue', label=str(conf)+'% CL')
    for i in range(Nmc):
        ax1.plot(MC_eng, MC_model[i,:], ls='-', linewidth=0.5, alpha=0.1, color='blue')

    ax1.errorbar(data['e_ref'], data['e2dnde'], yerr=data['e2dnde_err'],
                 marker='o', elinewidth=2, color='red',
                 markeredgecolor="black", markerfacecolor="red",
                 ls ='', label='Data')
    ax1.set_ylabel('$E^2 \\frac{dN}{dEdSdt}$ (MeV/cm$^2$/s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_xticks([])
    ax1.legend()
    
    # Add extra unit axes
    ax2 = ax1.twinx()
    ax2.plot(MC_eng, (bf_model2*u.Unit('MeV cm-2 s-1')).to_value('erg cm-2 s-1'), 'k-', alpha=0.0)
    ax2.set_ylabel('$E^2 \\frac{dN}{dEdSdt}$ (erg/cm$^2$/s)')
    ax2.set_yscale('log')
    ax2.set_ylim((ylim[0]*u.Unit('MeV cm-2 s-1')).to_value('erg cm-2 s-1'),
                 (ylim[1]*u.Unit('MeV cm-2 s-1')).to_value('erg cm-2 s-1'))
    
    # Residual plot
    ax3.plot(data['e_ref'], (data['e2dnde']-bf_model1)/data['e2dnde_err'],
             linestyle='', marker='o', color='k')
    ax3.plot(MC_eng,  MC_eng*0, linestyle='-', color='k')
    ax3.plot(MC_eng,  MC_eng*0+2, linestyle='--', color='k')
    ax3.plot(MC_eng,  MC_eng*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('Energy (GeV)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(cluster_test.output_dir+'/Ana_MCMC_spectrum_fitplot.pdf')
    plt.close()

    
#==================================================
# Get models from the parameter space
#==================================================

def get_mc_model(eng_mc, param_chains, cluster_test, Nmc=100):
    """
    Get models randomly sampled from the parameter space
        
    Parameters
    ----------
    - eng_mc (array): an array of energy
    - param_chains (ndarray): array of chains parametes
    - Nmc (int): number of models

    Output
    ------
    MC_model (ndarray): Nmc x N_eng array

    """

    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1],
                                    param_chains.shape[2])

    #----- Remove negative values (should never happen but in case)
    wneg = par_flat[:,0]>0
    par_flat = par_flat[wneg]
    
    Nsample = len(par_flat[:,0])-1
    
    MC_model = np.zeros((Nmc, len(eng_mc)))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        MC_model[i,:] = model_dNdEdSdt(cluster_test, eng_mc, param_MC)
    
    return MC_model


#==================================================
# Read the data
#==================================================

def read_data(specfile):
    """
    Read the data to extract the necessary information
    
    Parameters
    ----------
    - specfile (str): file where the data is stored

    Output
    ------
    - data (Table): Table containing the data

    """

    hdu = fits.open(specfile)
    spectrum = hdu[1].data
    hdu.close()

    e_ref         = spectrum['e_ref']*u.TeV
    e_min         = spectrum['e_min']*u.TeV
    e_max         = spectrum['e_max']*u.TeV
    npred         = spectrum['norm']*spectrum['ref_npred']
    e2dnde        = spectrum['norm']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
    dnde          = spectrum['norm']*spectrum['ref_dnde']/u.MeV/u.cm**2/u.s
    e2dnde_err    = spectrum['norm_err']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
    dnde_err      = spectrum['norm_err']*spectrum['ref_dnde']/u.MeV/u.cm**2/u.s
    e2dnde_ul     = spectrum['norm_ul']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
    dnde_ul       = spectrum['norm_ul']*spectrum['ref_dnde']/u.MeV/u.cm**2/u.s
    TS            = spectrum['ts']
    norm_scan     = spectrum['norm_scan']
    loglike       = spectrum['loglike']
    dloglike_scan = spectrum['dloglike_scan']

    # Fill my data
    data = Table()
    data['e_ref']         = e_ref.to_value('GeV')
    data['e_min']         = e_min.to_value('GeV') # Emin_bin = Eref-Emin
    data['e_max']         = e_max.to_value('GeV') # Emin_bin = Eref+Emax
    data['ref_dnde']      = (spectrum['ref_dnde']/u.MeV/u.cm**2/u.s).to_value('MeV-1 cm-2 s-1')
    data['ref_e2dnde']    = (spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s).to_value('MeV cm-2 s-1')
    data['dnde']          = dnde.to_value('MeV-1 cm-2 s-1')
    data['dnde_err']      = dnde_err.to_value('MeV-1 cm-2 s-1')
    data['e2dnde']        = e2dnde.to_value('MeV cm-2 s-1')
    data['e2dnde_err']    = e2dnde_err.to_value('MeV cm-2 s-1')
    data['norm_scan']     = norm_scan
    data['dloglike_scan'] = dloglike_scan

    # Warning if the error bars look weird
    TSbis = spectrum['norm']/spectrum['norm_err']
    test = TSbis/TS**0.5
    w_suspect = np.where(test > 3)[0]
    if len(w_suspect) > 0:
        print('-----> WARNING: some error bars are likely to be highly underestimated')
        print('       Index are ', w_suspect)
        print('       The errors are replaced by TS^{1/2}')
        print('       This does not affect if the likelihood scan is used (only the plot)')
        e2dnde_err_bis = spectrum['norm']*spectrum['ref_e2dnde'] / TS**0.5
        dnde_err_bis   = spectrum['norm']*spectrum['ref_dnde']   / TS**0.5
        unit = (1*u.erg/u.cm**2/u.s).to_value('MeV cm-2 s-1')        
        data['e2dnde_err'][w_suspect] = e2dnde_err_bis[w_suspect] * unit
        data['dnde_err'][w_suspect]   = dnde_err_bis[w_suspect]   * unit

    # Warning if likelihood scan is weird
    for i in range(dloglike_scan.shape[0]):
        if np.amax(dloglike_scan[i,:]) - np.amin(dloglike_scan[i,:]) < 1:
            print('WARNING: the likelihood scan in bin '+str(i)+' stays constant!')
            print('         this bin will be excluded')
            
    return data

    
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

def lnlike(params, cluster, data, par_min, par_max,
           gauss=True):
    '''
    Return the log likelihood for the given parameters

    Parameters
    ----------
    - params (list): the parameters
    - cluster (ClusterModel object): cluster test modeling object
    - data (Table): the data flux and errors
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
        
    test_model = model_dNdEdSdt(cluster, data['e_ref'], params)
    
    #---------- Compute the Gaussian likelihood
    # Gaussian likelihood
    if gauss:
        chi2 = (data['e2dnde'] - test_model)**2 / data['e2dnde_err']**2
        lnL = -0.5*np.nansum(chi2)
        
    # Likelihood taking into account true bin lnL
    else:
        Nbin = len(data['e_ref'])
        lnL_i = np.zeros(Nbin)
        for i in range(Nbin):
            # Check the the likelihood scan is ok (some scans stays there)
            cond = (np.amax(data['dloglike_scan'][i,:])-np.amin(data['dloglike_scan'][i,:])) > 1.0
            if cond: 
                # Interpolate the likelihood scan at the location of the model flux
                f = interp1d(data['norm_scan'][i,:]*data['ref_e2dnde'][i], data['dloglike_scan'][i,:],
                             fill_value='extrapolate') # extrapolate tested to work well on few sample
                lnL_i[i] = f(test_model[i])
            else:
                lnL_i[i] = 0.0
                
        lnL = np.sum(lnL_i)

    # In case of NaN, goes to infinity
    if np.isnan(lnL):
        lnL = -np.inf
        
    return lnL + prior


#==================================================
# MCMC: Defines model
#==================================================

def model_dNdEdSdt(cluster, energy, params):
    '''
    Gamma ray model for the MCMC

    Parameters
    ----------
    - cluster (ClusterModel object): cluster modeling object
    - energy (array, GeV): energy at which to compute the model
    - param (list): the parameter to sample in the model

    Output
    ------
    - output_model (array): the output model in MeV /cm2 / s
    '''
    
    #---------- Change parameters here
    cluster.X_crp_E            = {'X':params[0], 'R_norm':cluster.R500}
    cluster.spectrum_crp_model = {'name':'PowerLaw', 'Index':params[1]}
    
    #---------- Run the test model computation
    eng, dNdEdSdt = cluster.get_gamma_spectrum(energy*u.GeV,
                                                     Rmin=None, Rmax=cluster.R_truncation,
                                                     Rmin_los=None, NR500_los=5.0,
                                                     type_integral='spherical')
    
    output_model = (eng**2 * dNdEdSdt).to_value('MeV cm-2 s-1')
    
    return output_model


#==================================================
# MCMC: run the fit
#==================================================

def run_constraint(cluster_test,
                   spectrum_file,
                   nwalkers=10,
                   nsteps=1000,
                   burnin=100,
                   conf=68.0,
                   Nmc=100,
                   GaussLike=False,
                   reset_mcmc=False,
                   run_mcmc=True,
                   Emin=50,
                   Emax=10e3):
    """
    Run the MCMC constraints to the spectrum
        
    Parameters
    ----------
    - cluster_test (minot object): a cluster to be used when sampling parameters of model
    - spectrum_file (str): full path to the spectrum data
    - nwalkers (int): number of emcee wlakers
    - nsteps (int): number of emcee MCMC steps
    - burnin (int): number of point to remove assuming it is burnin
    - conf (float): confidence limit percentage for results
    - Nmc (int): number of monte carlo point when resampling the chains
    - GaussLike (bool): use gaussian approximation of the likelihood
    - reset_mcmc (bool): reset the existing MCMC chains?
    - run_mcmc (bool): run the MCMC sampling?                            
    - Emin/Emax (flaot, GeV): Energy min and max for flux/luminosity computation

    Output
    ------
    The final MCMC chains and plots are saved
    """

    #---------- Reset matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    #========== Guess parameter definition
    par0 = np.array([cluster_test.X_crp_E['X'],
                     cluster_test.spectrum_crp_model['Index']])
    
    parname = ['X_{CRp}', '\\alpha_{CRp}']
    par_min = [0.0,    2.0]
    par_max = [np.inf, 5.0]

    #========== Names
    sampler_file   = cluster_test.output_dir+'/Ana_MCMC_spectrum_sampler.pkl'
    chainstat_file = cluster_test.output_dir+'/Ana_MCMC_spectrum_chainstat.txt'
    chainplot_file = cluster_test.output_dir+'/Ana_MCMC_spectrum'
    global_file    = cluster_test.output_dir+'/Ana_MCMC_spectrum_globalprop.txt'

    #========== Start running MCMC definition and sampling
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(sampler_file)
    if sampler_exist:
        sampler = mcmc_common.load_object(sampler_file)
        print('    Existing sampler: '+sampler_file)
        
    #---------- Read the data
    data = read_data(spectrum_file)
    
    #---------- MCMC parameters
    ndim = len(par0)
    
    print('--- MCMC spectrum parameters: ')
    print('    Ndim                = '+str(ndim))
    print('    Nwalkers            = '+str(nwalkers))
    print('    Nsteps              = '+str(nsteps))
    print('    burnin              = '+str(burnin))
    print('    conf                = '+str(conf))
    print('    reset mcmc          = '+str(reset_mcmc))
    print('    Gaussian likelihood = '+str(GaussLike))

    #---------- Defines the start
    if sampler_exist:
        if reset_mcmc:
            print('--- Reset MCMC even though sampler already exists')
            pos = mcmc_common.chains_starting_point(par0, 0.1, par_min, par_max, nwalkers)
            sampler.reset()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                            args=[cluster_test, data, par_min, par_max, GaussLike])
        else:
            print('--- Start from already existing sampler')
            pos = sampler.chain[:,-1,:]
    else:
        print('--- No pre-existing sampler, start from scratch')
        pos = mcmc_common.chains_starting_point(par0, 0.1, par_min, par_max, nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                        args=[cluster_test, data, par_min, par_max, GaussLike])
        
    #---------- Run the MCMC
    if run_mcmc:
        print('--- Runing '+str(nsteps)+' MCMC steps')
        sampler.run_mcmc(pos, nsteps, progress=True)

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
    MC_eng     = np.logspace(-1, 5, 50) # GeV
    MC_model   = get_mc_model(MC_eng, param_chains, cluster_test, Nmc=Nmc)
    Best_model = model_dNdEdSdt(cluster_test, MC_eng, par_best)

    #---------- Plots and results
    mcmc_common.chains_plots(param_chains, parname, chainplot_file,
                             par_best=par_best, par_percentile=par_percentile, conf=conf,
                             par_min=par_min, par_max=par_max)
    
    get_global_prop(MC_eng, MC_model, Best_model, cluster_test.D_lum.to_value('cm'),
                    chainplot_file,
                    Emin=Emin, Emax=Emax, # GeV
                    outfile=global_file)

    modelplot(data, cluster_test, par_best, param_chains, MC_eng, MC_model, conf=conf)
