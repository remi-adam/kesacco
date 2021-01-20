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
import seaborn as sns
import pandas as pd
import emcee
import corner

from minot.model_tools import trapz_loglog
from minot.ClusterTools import map_tools
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
                file.write('  median = '+str(perc[1])+' -'+str(perc[1]-perc[0])+' +'+str(perc[2]-perc[1])+'\n')
                file.write('  best   = '+str(par_best[ipar])+' -'+str(par_best[ipar]-perc[0])+' +'+str(perc[2]-par_best[ipar])+'\n')
                file.write('   '+parnamei+' = '+txt+'\n')

    if outfile is not None:
        file.close() 
            
    return par_best, par_percentile


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
              theta=1*u.deg):
    """
    Plot the data versus model and constraints
        
    Parameters
    ----------

    Output
    ------
    Plots are saved
    """
    
    reso = header['CDELT2']
    sigma_sm = (FWHM/(2*np.sqrt(2*np.log(2)))).to_value('deg')/reso
    
    #========== Data - model, stack
    fig = plt.figure(0, figsize=(15, 4))
    ax = plt.subplot(131, projection=WCS(header), slices=('x', 'y', 0))
    plt.imshow(gaussian_filter(np.sum(data, axis=0), sigma=sigma_sm),
               origin='lower', cmap='magma',norm=SymLogNorm(1))
    cb = plt.colorbar()
    plt.title('Data (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    
    ax = plt.subplot(132, projection=WCS(header), slices=('x', 'y', 0))
    plt.imshow(gaussian_filter(np.sum(modbest['cluster']+modbest['background'],axis=0), sigma=sigma_sm),
               origin='lower', cmap='magma', vmin=cb.norm.vmin, vmax=cb.norm.vmax, norm=SymLogNorm(1))
    plt.colorbar()
    plt.title('Model (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    
    ax = plt.subplot(133, projection=WCS(header), slices=('x', 'y', 0))
    plt.imshow(gaussian_filter(np.sum(data-(modbest['cluster']+modbest['background']), axis=0), sigma=sigma_sm),
               origin='lower', cmap='RdBu')
    plt.colorbar()
    plt.title('Residual (counts)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    
    plt.savefig(outdir+'/MCMC_MapResidual.pdf')
    plt.close()

    #========== Data - model, for all energy bins
    pdf_pages = PdfPages(outdir+'/MCMC_MapSliceResidual.pdf')
    
    for i in range(len(Ebins)):
        fig = plt.figure(0, figsize=(15, 4))
        ax = plt.subplot(131, projection=WCS(header), slices=('x', 'y', i))
        plt.imshow(gaussian_filter(data[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='magma', norm=SymLogNorm(1))
        cb = plt.colorbar()
        plt.title('Data (counts) - E=['+'{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        
        ax = plt.subplot(132, projection=WCS(header), slices=('x', 'y', i))
        plt.imshow(gaussian_filter((modbest['cluster']+modbest['background'])[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='magma',vmin=cb.norm.vmin, vmax=cb.norm.vmax, norm=SymLogNorm(1))
        plt.colorbar()
        plt.title('Model (counts) - E=['+'{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')
        
        ax = plt.subplot(133, projection=WCS(header), slices=('x', 'y', 0))
        plt.imshow(gaussian_filter((data-(modbest['cluster']+modbest['background']))[i,:,:], sigma=sigma_sm),
                   origin='lower', cmap='RdBu')
        plt.colorbar()
        plt.title('Residual (counts) - E=['+'{:.1f}'.format(Ebins[i][0]*1e-6)+', '+'{:.1f}'.format(Ebins[i][1]*1e-6)+'] GeV')
        plt.xlabel('R.A.')
        plt.ylabel('Dec.')

        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()
    
    #========== Spectrum within theta
    #----- Compute a mask
    header2 = copy.copy(header)
    header2['NAXIS'] = 2
    del header2['NAXIS3']
    ra_map, dec_map = map_tools.get_radec_map(header2)
    radmap = map_tools.greatcircle(ra_map, dec_map, np.median(ra_map), np.median(dec_map))
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
    cluster_mc_spec    = np.zeros((MC_model['cluster'].shape[0], len(Ebins)))
    background_mc_spec = np.zeros((MC_model['cluster'].shape[0], len(Ebins)))
    tot_mc_spec        = np.zeros((MC_model['cluster'].shape[0], len(Ebins)))

    for i in range(MC_model['cluster'].shape[0]):
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
    
    #----- Figure
    fig = plt.figure(1, figsize=(8, 6))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(Emean, data_spec, yerr=np.sqrt(data_spec),
                 xerr=[Emean-Ebins['E_MIN'], Ebins['E_MAX']-Emean],fmt='ko', capsize=0, linewidth=2, zorder=2, label='Data')
    plt.step(binsteps, np.append(cluster_spec,cluster_spec[-1]),
             where='post', color='blue', linewidth=2, label='Cluster model')
    plt.step(binsteps, np.append(background_spec, background_spec[-1]),
             where='post', color='red', linewidth=2, label='Background model')
    plt.step(binsteps, np.append(cluster_spec+background_spec, (cluster_spec+background_spec)[-1]),
             where='post', color='green', linewidth=2, label='Total model')

    plt.step(binsteps, np.append(cluster_up_spec, cluster_up_spec[-1]),
             where='post', color='blue', linewidth=1, linestyle='--')
    plt.step(binsteps, np.append(cluster_lo_spec, cluster_lo_spec[-1]),
             where='post', color='blue', linewidth=1, linestyle='--')
    plt.step(binsteps, np.append(background_up_spec, background_up_spec[-1]),
             where='post', color='red', linewidth=1, linestyle='--')
    plt.step(binsteps, np.append(background_lo_spec, background_lo_spec[-1]),
             where='post', color='red', linewidth=1, linestyle='--')
    plt.step(binsteps, np.append(tot_lo_spec, tot_lo_spec[-1]),
             where='post', color='green', linewidth=1, linestyle='--')
    plt.step(binsteps, np.append(tot_up_spec, tot_up_spec[-1]),
             where='post', color='green', linewidth=1, linestyle='--')
    plt.fill_between(Emean, cluster_up_spec, cluster_lo_spec, alpha=0.3, color='blue')
    plt.fill_between(Emean, background_up_spec, background_lo_spec, alpha=0.3, color='red')
    plt.fill_between(Emean, tot_up_spec, tot_lo_spec, alpha=0.3, color='green')
    
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(binsteps), np.amax(binsteps))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.legend()
    plt.title('Spectrum within $\\theta = $'+str(theta))

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.plot(Emean, (data_spec-cluster_spec-background_spec)/np.sqrt(data_spec), marker='o', color='k', linestyle='')
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
    - specfile (str): file where the data is stored

    Output
    ------
    - data (Table): Table containing the data

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
    models_cl  = hdu[3].data
    models_bk  = hdu[4].data
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
        chi2 = (data - test_model['cluster'] - test_model['background'])**2 / np.sqrt(test_model['cluster'])**2
        lnL = -0.5*np.nansum(chi2)

    # Poisson with Bkg
    else:        
        L_i = test_model['cluster']+test_model['background'] - data*np.log(test_model['cluster']+test_model['background'])
        lnL  = -np.nansum(L_i)
        
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
                   run_mcmc=True):
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
    parname = ['X_{CRp}/X_{CRp, input}', '\\eta_{CRp}', '\\alpha_{CRp}'] 
    par0 = np.array([1.0, np.mean(modgrid['spa_val']), np.mean(modgrid['spe_val'])])
    par_min = [0,      np.amin(modgrid['spa_val']), np.amin(modgrid['spe_val'])]
    par_max = [np.inf, np.amax(modgrid['spa_val']), np.amax(modgrid['spe_val'])]
    
    #========== Start running MCMC definition and sampling    
    #---------- Check if a MCMC sampler was already recorded
    sampler_exist = os.path.exists(subdir+'/MCMC_sampler.pkl')
    if sampler_exist:
        sampler = load_object(subdir+'/MCMC_sampler.pkl')
        print('    Existing sampler: '+subdir+'/MCMC_sampler.pkl')
    
    #---------- MCMC parameters
    ndim = len(par0)
    
    print('--- MCMC profile parameters: ')
    print('    ndim       = '+str(ndim))
    print('    nwalkers   = '+str(nwalkers))
    print('    nsteps     = '+str(nsteps))
    print('    burnin     = '+str(burnin))
    print('    conf       = '+str(conf))
    print('    reset_mcmc = '+str(reset_mcmc))
    print('    Gaussian L = '+str(GaussLike))

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
    save_object(sampler, subdir+'/MCMC_sampler.pkl')

    #---------- Burnin
    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains = sampler.lnprobability[:, burnin:]
    
    #---------- Get the parameter statistics
    par_best, par_percentile = chains_statistics(param_chains, lnL_chains,
                                                 parname=parname, conf=conf, show=True,
                                                 outfile=subdir+'/MCMC_chainstat.txt')
    
    #---------- Get the well-sampled models
    MC_model   = get_mc_model(modgrid, param_chains, Nmc=Nmc)
    Best_model = model_specimg(modgrid, par_best)

    #---------- Plots and results
    chainplots(param_chains, parname, subdir+'/MCMC_chainplot',
               par_best=par_best, par_percentile=par_percentile, conf=conf,
               par_min=par_min, par_max=par_max)
    
    modelplot(data, Best_model, MC_model, modgrid['header'], modgrid['Ebins'], subdir,
              conf=conf, FWHM=0.1*u.deg, theta=1.0*u.deg)
