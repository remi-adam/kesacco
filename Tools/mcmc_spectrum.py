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

    #----- Remove negative values (should never happen but does)
    wneg = par_flat[:,0]>0
    par_flat = par_flat[wneg]
    
    Nsample = len(par_flat[:,0])-1
    
    MC_model = np.zeros((Nmc, len(eng_mc)))
    
    for i in range(Nmc):
        param_MC = par_flat[np.random.randint(0, high=Nsample), :] # randomly taken from chains
        MC_model[i,:] = model_dNdEdSdt(cluster_test, eng_mc, param_MC)
    
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
        fig = plt.figure(ipar, figsize=(8, 6))
        ax = sns.distplot(param_chains[:,:,ipar].flatten(), bins=40, kde=True, color='blue')
        ymax = ax.get_ylim()[1]
        if par_best is not None:
            ax.vlines(par_best[ipar], 0, ymax, linestyle='-', label='Maximum likelihood')
        if par_percentile is not None:
            ax.vlines(par_percentile[1,ipar], 0.0, ymax, linestyle='--', label='Median')
            ax.vlines(par_percentile[0,ipar], 0.0, ymax, linestyle=':', color='orange')
            ax.vlines(par_percentile[2,ipar], 0.0, ymax, linestyle=':', color='orange')
            ax.fill_between(par_percentile[:,ipar], [0,0,0], y2=[ymax,ymax,ymax],
                       alpha=0.2, color='orange', label=str(conf)+'% CL')
        ax.set_xlabel('$'+parname[ipar]+'$')
        ax.set_ylabel('$P('+parname[ipar]+')$')
        ax.set_yticks([])
        if par_min is not None and par_max is not None:
            xlim = [ax.get_xlim()[0], ax.get_xlim()[1]]
            if ax.get_xlim()[0] < par_min[ipar]:
                xlim[0] = par_min[ipar]
            if ax.get_xlim()[1] > par_max[ipar]:
                xlim[1] = par_max[ipar]
            ax.set_xlim(xlim)
        ax.set_ylim(0,ymax)
        ax.legend()
        fig.savefig(rout_file+'_MCMC_chain_histo'+str(ipar)+'.pdf')
        plt.close()

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
                            n_levels=30, cols=[('green', 'k', 'grey', 'YlGn')], 
                            perc=[0.68, 0.95], gridsize=100,
                            linewidth=2.0, alpha=(0.3, 1.0), figsize=((Npar+1)*3,(Npar+1)*3))
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
    fig = plt.figure(figsize=(8, 6))
    ax = sns.distplot(F1_mc, bins=40, kde=True, color='blue')
    ymax = ax.get_ylim()[1]
    ax.vlines(F1_B, 0, ymax, linestyle='-', label='Maximum likelihood')
    ax.vlines(F1_perc[1], 0.0, ymax, linestyle='--', label='Median')
    ax.vlines(F1_perc[0], 0.0, ymax, linestyle=':', color='orange')
    ax.vlines(F1_perc[2], 0.0, ymax, linestyle=':', color='orange')
    ax.fill_between(F1_perc, [0,0,0], y2=[ymax,ymax,ymax],
                    alpha=0.2, color='orange', label=str(conf)+'% CL')
    ax.set_xlabel('Flux (ph s$^{-1}$ cm$^{-2}$)')
    ax.set_ylabel('$P(Flux)$')
    ax.set_yticks([])
    #if par_min is not None and par_max is not None:
    #    xlim = [ax.get_xlim()[0], ax.get_xlim()[1]]
    #    if ax.get_xlim()[0] < par_min[ipar]:
    #        xlim[0] = par_min[ipar]
    #    if ax.get_xlim()[1] > par_max[ipar]:
    #        xlim[1] = par_max[ipar]
    #        ax.set_xlim(xlim)
    ax.set_ylim(0,ymax)
    ax.legend()
    fig.savefig(rout_file+'_MCMC_chain_histo_Flux.pdf')
    plt.close()

    
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
    bf_model1 = model_dNdEdSdt(cluster_test, data['energy'], par_best)
    bf_model2 = model_dNdEdSdt(cluster_test, MC_eng, par_best)
    MC_perc   = np.percentile(MC_model, [(100-conf)/2.0, 50, 100 - (100-conf)/2.0], axis=0)
    
    #========== Plot
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    xlim = [np.amin(MC_eng), np.amax(MC_eng)]
    rngyp1 = 1.2*np.nanmax(data['flux']+data['e_flux'])
    rngyp2 = 1.2*np.nanmax(bf_model2)
    rngyp  = np.amax(np.array([rngyp1,rngyp2]))
    rngym = 0.5*np.nanmin(data['flux'])
    ylim = [rngym, rngyp]

    ax1.plot(MC_eng, bf_model2,    ls='-', linewidth=2, color='k', label='Maximum likelihood model')
    ax1.plot(MC_eng, MC_perc[1,:], ls='--', linewidth=2, color='b', label='Median')
    ax1.plot(MC_eng, MC_perc[0,:], ls=':', linewidth=1, color='b')
    ax1.plot(MC_eng, MC_perc[2,:], ls=':', linewidth=1, color='b')
    ax1.fill_between(MC_eng, MC_perc[0,:], y2=MC_perc[2,:], alpha=0.2, color='blue', label=str(conf)+'% CL')
    for i in range(Nmc):
        ax1.plot(MC_eng, MC_model[i,:], ls='-', linewidth=0.5, alpha=0.1, color='blue')

    ax1.errorbar(data['energy'], data['flux'], yerr=data['e_flux'],
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
    ax3.plot(data['energy'], (data['flux']-bf_model1)/data['e_flux'],
             linestyle='', marker='o', color='k')
    ax3.plot(MC_eng,  MC_eng*0, linestyle='-', color='k')
    ax3.plot(MC_eng,  MC_eng*0+2, linestyle='--', color='k')
    ax3.plot(MC_eng,  MC_eng*0-2, linestyle='--', color='k')
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(-3, 3)
    ax3.set_xlabel('Energy (GeV)')
    ax3.set_ylabel('$\\chi$')
    ax3.set_xscale('log')
    
    fig.savefig(cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_results.pdf')
    plt.close()


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

    # the data content depends on the version
    try:
        energy     = spectrum['e_ref']*u.TeV
        ed_Energy  = spectrum['e_min']*u.TeV
        eu_Energy  = spectrum['e_min']*u.TeV
        npred      = spectrum['norm']*spectrum['ref_npred']
        flux       = spectrum['norm']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
        e_flux     = spectrum['norm_err']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
        UpperLimit = spectrum['norm_ul']*spectrum['ref_e2dnde']*u.erg/u.cm**2/u.s
        TS         = spectrum['ts']
    except:
        energy     = spectrum['Energy']*u.TeV
        ed_Energy  = spectrum['ed_Energy']*u.TeV
        eu_Energy  = spectrum['eu_Energy']*u.TeV
        npred      = spectrum['Npred']
        flux       = spectrum['Flux']*u.erg/u.cm**2/u.s
        e_flux     = spectrum['e_Flux']*u.erg/u.cm**2/u.s
        UpperLimit = spectrum['UpperLimit']*u.erg/u.cm**2/u.s
        TS         = spectrum['TS']

    # Fill my data
    data = Table()
    data['energy'] = energy.to_value('GeV')
    data['flux']   = flux.to_value('MeV cm-2 s-1')
    data['e_flux'] = e_flux.to_value('MeV cm-2 s-1')
    
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

def lnlike(params, cluster, data, par_min, par_max):
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
        
    test_model = model_dNdEdSdt(cluster, data['energy'], params)
    
    #---------- Compute the Gaussian likelihood
    chi2 = (data['flux'] - test_model)**2 / data['e_flux']**2
    lnL = -0.5*np.nansum(chi2)
    
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
    eng, model_dNdEdSdt = cluster.get_gamma_spectrum(energy*u.GeV,
                                                     Rmin=None, Rmax=cluster.R_truncation,
                                                     Rmin_los=None, NR500_los=5.0,
                                                     type_integral='spherical')
    
    output_model = (eng**2 * model_dNdEdSdt).to_value('MeV cm-2 s-1')
    
    return output_model


#==================================================
# MCMC: run the fit
#==================================================

def run_spectrum_constraint(cluster_test,
                            spectrum_file,
                            nwalkers=10,
                            nsteps=1000,
                            burnin=100,
                            conf=68.0,
                            Nmc=100,
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
    par_min = [0.0,                         2.0]
    par_max = [5*cluster_test.X_crp_E['X'], 2*cluster_test.spectrum_crp_model['Index']]
    
    #========== Start running MCMC definition and sampling
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_sampler.pkl')
    if sampler_exist:
        sampler = load_object(cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_sampler.pkl')
        print('    Existing sampler: '+cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_sampler.pkl')
        
    #---------- Read the data
    data = read_data(spectrum_file)
    
    #---------- MCMC parameters
    ndim = len(par0)
    
    print('--- MCMC spectrum parameters: ')
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
                                            args=[cluster_test, data, par_min, par_max])
        else:
            print('--- Start from already existing sampler')
            pos = sampler.chain[:,-1,:]
    else:
        print('--- No pre-existing sampler, start from scratch')
        pos = [par0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
        #pos = [np.random.uniform(par_min[i],par_max[i], nwalkers) for i in range(ndim)]
        #pos = list(np.array(pos).T.reshape((nwalkers,ndim)))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                        args=[cluster_test, data, par_min, par_max])

    #---------- Run the MCMC
    if run_mcmc:
        print('--- Runing '+str(nsteps)+' MCMC steps')
        sampler.run_mcmc(pos, nsteps)

    #---------- Save the MCMC after the run
    save_object(sampler, cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_sampler.pkl')

    #---------- Burnin
    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains = sampler.lnprobability[:, burnin:]

    #---------- Get the parameter statistics
    par_best, par_percentile = chains_statistics(param_chains, lnL_chains,
                                                 parname=parname, conf=conf, show=True,
                                                 outfile=cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_chainstat.txt')
    
    #---------- Get the well-sampled models
    #MC_eng     = np.logspace(np.log10(np.amin(data['energy'])/2),
    #                         np.log10(np.amax(data['energy'])*2.0), 20)
    MC_eng     = np.logspace(-1, 6, 50) # GeV
    MC_model   = get_mc_model(MC_eng, param_chains, cluster_test, Nmc=Nmc)
    Best_model = model_dNdEdSdt(cluster_test, MC_eng, par_best)

    #---------- Plots and results
    chainplots(param_chains, parname, cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name,
               par_best=par_best, par_percentile=par_percentile, conf=conf,
               par_min=par_min, par_max=par_max)
    
    get_global_prop(MC_eng, MC_model, Best_model, cluster_test.D_lum.to_value('cm'),
                    cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name,
                    Emin=Emin, Emax=Emax, # GeV
                    outfile=cluster_test.output_dir+'/Ana_spectrum_'+cluster_test.name+'_MCMC_globalprop.txt')

    modelplot(data, cluster_test, par_best, param_chains, MC_eng, MC_model, conf=conf)
