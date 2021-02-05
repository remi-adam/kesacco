"""
This file contains various functions associated to MCMC analysis
that are common to the different implemented methods

"""

#==================================================
# Requested imports
#==================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner

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

def chains_statistics(param_chains,
                      lnL_chains,
                      parname=None,
                      conf=68.0,
                      show=True,
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
            
            medval = str(perc[1])+' -'+str(perc[1]-perc[0])+' +'+str(perc[2]-perc[1])
            bfval = str(par_best[ipar])+' -'+str(par_best[ipar]-perc[0])+' +'+str(perc[2]-par_best[ipar])

            print('param '+str(ipar)+' ('+parnamei+'): ')
            print('   median   = '+medval)
            print('   best-fit = '+bfval)
            print('   '+parnamei+' = '+txt)

            if outfile is not None:
                file.write('param '+str(ipar)+' ('+parnamei+'): '+'\n')
                file.write('  median = '+medval+'\n')
                file.write('  best   = '+bfval+'\n')
                file.write('   '+parnamei+' = '+txt+'\n')

    if outfile is not None:
        file.close() 
            
    return par_best, par_percentile


#==================================================
# Plots related to the chains
#==================================================

def chains_plots(param_chains,
                 parname,
                 rout_file,
                 par_best=None,
                 par_percentile=None,
                 conf=68.0,
                 par_min=None,
                 par_max=None):
    """
    Plot related to MCMC chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - rout_file (str): root file where to save plots, e.g. directory+'/MCMC'
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
                            output_fig=rout_file+'_histo'+str(ipar)+'.pdf',
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
    fig.savefig(rout_file+'_chains.pdf')
    plt.close()

    # Corner plot using seaborn
    parname_corner = []
    for i in range(Npar): parname_corner.append('$'+parname[i]+'$')
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
    df = pd.DataFrame(par_flat, columns=parname_corner)
    plotting.seaborn_corner(df, output_fig=rout_file+'_triangle_seaborn.pdf',
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
    figure.savefig(rout_file+'_triangle_corner.pdf')
    plt.close("all")

    
#==================================================
# Starting point
#==================================================

def chains_starting_point(guess, disp, par_min, par_max, nwalkers):
    """
    Sample the parameter space for chain starting point
        
    Parameters
    ----------
    - guess (Nparam array): guess value of the parameters
    - disp (float): dispersion allowed for unborned parameters
    - parmin (Nparam array): min value of the parameters
    - parmax (Nparam array): max value of the parameters
    - nwalkers (int): number of walkers

    Output
    ------
    start: the starting point of the chains in the parameter space

    """

    ndim = len(guess)

    # First range using guess + dispersion
    vmin = guess - guess*disp 
    vmax = guess + guess*disp

    # If parameters are 0, uses born
    w0 = np.where(guess < 1e-4)[0]
    for i in range(len(w0)):
        if np.array(par_min)[w0[i]] != np.inf and np.array(par_min)[w0[i]] != -np.inf:
            vmin[w0[i]] = np.array(par_min)[w0[i]]
            vmax[w0[i]] = np.array(par_max)[w0[i]]
        else:
            vmin[w0[i]] = -1.0
            vmax[w0[i]] = +1.0
            print('Warning, some starting point parameters are difficult to estimate')
            print('because the guess parameter is 0 and the par_min/max are infinity')

    # Check that the parameters are in the prior range
    wup = vmin < np.array(par_min)
    wlo = vmax > np.array(par_max)
    vmin[wup] = np.array(par_min)[wup]
    vmax[wlo] = np.array(par_max)[wlo]

    # Get parameters
    start = [np.random.uniform(low=vmin, high=vmax) for i in range(nwalkers)]

    #print(guess)
    #print(par_min)
    #print(par_max)
    #print(vmin)
    #print(vmax)
    
    return start
