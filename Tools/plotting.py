"""
This file contains general plotting tools in common 
with Ana and Sim modules.

"""

#==================================================
# Requested imports
#==================================================

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy import interpolate
import scipy.ndimage as ndimage
import random

from kesacco.Tools import plotting_irf
from kesacco.Tools import plotting_obsfile

import gammalib

#==================================================
# Style
#==================================================

cta_energy_range   = [0.02, 100.0]*u.TeV
fermi_energy_range = [0.1, 300.0]*u.GeV

def set_default_plot_param(leftspace=0.18, rightspace=0.87):
    
    dict_base = {'font.size':        16, 
                 'legend.fontsize':  16,
                 'xtick.labelsize':  16,
                 'ytick.labelsize':  16,
                 'axes.labelsize':   16,
                 'axes.titlesize':   16,
                 'figure.titlesize': 16,
                 'figure.figsize':[8.0, 6.0],
                 'figure.subplot.right':rightspace,
                 'figure.subplot.left':leftspace, # Ensure enough space on the left so that all plot can be aligned
                 'font.family':'serif',
                 'figure.facecolor': 'white',
                 'legend.frameon': True}

    plt.rcParams.update(dict_base)

    
#==================================================
# Usefull plot functions
#==================================================

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    From https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
    if not ax:
        ax = plt.gca()
        
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=+30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


#==================================================
# Extract correlation matrix
#==================================================

def correlation_from_covariance(covariance):
    """
    Compute the correlation matrix from the covariance
    
    Parameters
    ----------
    - covariance (2d array): the covariance matrix

    Output
    ------
    - correlation (2d array): the correlation matrix

    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    
    return correlation

    
#==================================================
# Get the CTA PSF given the IRF
#==================================================

def get_cta_psf(caldb, irf, emin, emax, w8_slope=-2):
    """
    Return the on-axis maximum PSF between emin and emax.

    Parameters
    ----------
    - caldb (str): the calibration database
    - irf (str): input response function
    - emin (float): the minimum energy considered (TeV)
    - emax (float): the maximum energy considered (TeV)
    - w8_slope (float): spectral slope assumed to weight 
    the PSF when extracting the mean

    Outputs
    --------
    - PSF (FWHM, deg): on axis point spread function
    """

    CTOOLS_dir = os.getenv('CTOOLS')

    data_file = CTOOLS_dir+'/share/caldb/data/cta/'+caldb+'/bcf/'+irf+'/irf_file.fits'
    hdul = fits.open(data_file)
    data_PSF = hdul[2].data
    hdul.close()

    theta_mean = (data_PSF['THETA_LO'][0,:]+data_PSF['THETA_HI'][0,:])/2.0
    eng_mean = (data_PSF['ENERG_LO'][0,:]+data_PSF['ENERG_HI'][0,:])/2.0
    PSF_E = data_PSF['SIGMA_1'][0,0,:] # This is on axis
    
    fitpl  = interpolate.interp1d(eng_mean, PSF_E, kind='cubic')
    e_itpl = np.logspace(np.log10(emin), np.log10(emax), 1000)
    PSF_itpl = fitpl(e_itpl)
    PSF_itpl = PSF_itpl * 2*np.sqrt(2*np.log(2)) # Convert to FWHM
                     
    weng = (e_itpl > emin) * (e_itpl < emax)

    PSF = np.sum(PSF_itpl[weng] * e_itpl[weng]**w8_slope) / np.sum(e_itpl[weng]**w8_slope)
    
    return PSF


#==================================================
#Show the IRF
#==================================================

def show_irf(caldb_in, irf_in, plotfile,
             emin=None, emax=None,
             tmin=None, tmax=None):
    """
    Show the IRF by calling the ctools function

    Parameters
    ----------
    - caldb_in (str list): the calibration database
    - irf_in (str list): input response function
    - emin (min energy): minimal energy in TeV
    - emax (max energy): maximal energy in TeV
    - tmin (min energy): minimal angle in deg
    - tmax (max energy): maximal angle in deg

    """

    set_default_plot_param()

    # Select all the unique IRF
    list_use  = []
    caldb_use = []
    irf_use   = []
    for i in range(len(caldb_in)):
        if caldb_in[i] + irf_in[i] not in list_use:
            list_use.append(caldb_in[i] + irf_in[i])
            caldb_use.append(caldb_in[i])
            irf_use.append(irf_in[i])

    # ----- Loop over all caldb+irf used
    for i in range(len(caldb_use)):
           
        # Convert to gammalib format
        caldb = gammalib.GCaldb('cta', caldb_use[i])
        irf   = gammalib.GCTAResponseIrf(irf_use[i], caldb)

        # Build selection string
        selection  = ''
        eselection = ''
        tselection = ''
        if emin != None and emax != None:
            eselection += '%.3f-%.1f TeV' % (emin, emax)
        elif emin != None:
            eselection += ' >%.3f TeV' % (emin)
        elif emax != None:
            eselection += ' <%.1f TeV' % (emax)
        if tmin != None and tmax != None:
            tselection += '%.1f-%.1f deg' % (tmin, tmax)
        elif tmin != None:
            tselection += ' >%.1f deg' % (tmin)
        elif tmax != None:
            tselection += ' <%.1f deg' % (tmax)
        if len(eselection) > 0 and len(tselection) > 0:
            selection = ' (%s, %s)' % (eselection, tselection)
        elif len(eselection) > 0:
            selection = ' (%s)' % (eselection)
        elif len(tselection) > 0:
            selection = ' (%s)' % (tselection)

        # Build title
        mission    = irf.caldb().mission()
        instrument = irf.caldb().instrument()
        response   = irf.rspname()
        title      = '%s "%s" Instrument Response Function "%s"%s' % \
            (gammalib.toupper(mission), instrument, response, selection)

        # Create figure
        fig = plt.figure(figsize=(22,12))
        
        # Add title
        fig.suptitle(title, fontsize=16)
        
        # Plot Aeff
        ax1 = fig.add_subplot(231)
        plotting_irf.plot_aeff(ax1, irf.aeff(), emin=emin, emax=emax, tmin=tmin, tmax=tmax)
        
        # Plot Psf
        ax2 = fig.add_subplot(232)
        plotting_irf.plot_psf(ax2, irf.psf(), emin=emin, emax=emax, tmin=tmin, tmax=tmax)
        
        # Plot Background
        ax3 = fig.add_subplot(233)
        plotting_irf.plot_bkg(ax3, irf.background(), emin=emin, emax=emax, tmin=tmin, tmax=tmax)
        
        # Plot Edisp
        fig.add_subplot(234)
        plotting_irf.plot_edisp(irf.edisp(), emin=emin, emax=emax, tmin=tmin, tmax=tmax)
        
        # Show plots or save it into file
        plt.savefig(plotfile+'_'+list_use[i]+'.pdf')
        plt.close()

    return


#==================================================
# Show map
#==================================================

def show_map(mapfile, outfile,
             smoothing_FWHM=0.0*u.deg,
             cluster_ra=None,
             cluster_dec=None,
             cluster_t500=None,
             cluster_name='',
             ps_name=[],
             ps_ra=[],
             ps_dec=[],
             ptg_ra=None,
             ptg_dec=None,
             PSF=None,
             maptitle='',
             bartitle='',
             rangevalue=[None, None],
             logscale=True,
             significance=False,
             cmap='magma'):
    """
    Plot maps to show.

    Parameters
    ----------
    Mandatory parameters:
    - mapfile (str): the map fits file to use
    - outfile (str): the ooutput plot file

    Optional parameters:
    - smoothing_FWHM (angle unit): FWHM used for smoothing
    - cluster_ra,dec (deg) : the center of the cluster in RA Dec
    - cluster_t500 (deg): cluster theta 500
    - cluster_name (str): name of the cluster
    - ps_name (str list): list of point source names
    - ps_ra,dec (deg list): list of point source RA and Dec
    - ptg_ra,dec (deg): pointing RA, Dec
    - PSF (deg): the PSF FWHM in deg
    - maptitle (str): title
    - bartitle (str): title of the colorbar
    - rangevalue (float list): range of teh colorbar
    - logscale (bool): apply log color bar
    - significance (bool): is this a significance map?
    - cmap (str): colormap

    Outputs
    --------
    - validation plot map
    """

    set_default_plot_param()

    #---------- Read the data
    data = fits.open(mapfile)[0]
    image = data.data
    wcs_map = WCS(data.header)
    reso = abs(wcs_map.wcs.cdelt[0])
    Npixx = image.shape[0]
    Npixy = image.shape[1]
    fov_x = Npixx * reso
    fov_y = Npixy * reso
            
    #---------- Smoothing
    sigma_sm = (smoothing_FWHM/(2*np.sqrt(2*np.log(2)))).to_value('deg')/reso
    image = ndimage.gaussian_filter(image, sigma=sigma_sm)

    if significance:
        norm = 2*sigma_sm*np.sqrt(np.pi) # Mean noise smoothing reduction, assuming gaussian correlated noise
        image *= norm
        print('WARNING: The significance is boosted accounting for smoothing.')
        print('         This assumes weak noise spatial variarion (w.r.t. smoothing),')
        print('         gaussian regime, and uncorrelated pixels.')
        
    #--------- map range
    if rangevalue[0] is None:
        vmin = np.amin(image)
    else:
        vmin = rangevalue[0]

    if rangevalue[1] is None:
        vmax = np.amax(image)
    else:
        vmax = rangevalue[1]
        
    #---------- Plot
    if not ((np.amax(image) == 0) and (np.amin(image) == 0)) :
        fig = plt.figure(1, figsize=(12, 9))
        ax = plt.subplot(111, projection=wcs_map)

        if logscale :
            plt.imshow(image, origin='lower', cmap=cmap, norm=SymLogNorm(1), vmin=vmin, vmax=vmax)
        else:
            plt.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            
        # Show cluster t500
        if (cluster_t500 is not None) * (cluster_ra is not None) * (cluster_dec is not None) :
            circle_rad = 2*cluster_t500/np.cos(cluster_dec*np.pi/180)
            circle_500 = matplotlib.patches.Ellipse((cluster_ra, cluster_dec),
                                                    circle_rad,
                                                    2*cluster_t500,
                                                    linewidth=2, fill=False, zorder=2,
                                                    edgecolor='lightgray', linestyle='dashed',
                                                    facecolor='none',
                                                    transform=ax.get_transform('fk5'),
                                                    label='$R_{500}$')
            ax.add_patch(circle_500)
            txt_r500 = plt.text(cluster_ra - cluster_t500, cluster_dec - cluster_t500,
                                '$R_{500}$',
                                transform=ax.get_transform('fk5'), fontsize=10,
                                color='lightgray',
                                horizontalalignment='center',verticalalignment='center')

        # Show the pointing
        if (ptg_ra is not None) * (ptg_dec is not None) :
            ax.scatter(ptg_ra, ptg_dec,
                       transform=ax.get_transform('icrs'), color='gray', marker='+', s=100,
                       label='Pointings')

            try:
                txt_ptg = plt.text(ptg_ra, ptg_dec+0.2, 'Pointing',
                                   transform=ax.get_transform('fk5'),fontsize=10,
                                   color='gray', horizontalalignment='center',
                                   verticalalignment='center')
            except:
                txt_ptg = plt.text(ptg_ra[0], ptg_dec[0]+0.2, 'Pointings',
                                   transform=ax.get_transform('fk5'),fontsize=10,
                                   color='gray', horizontalalignment='center',
                                   verticalalignment='center')

                
        # Show the cluster center
        if (cluster_ra is not None) * (cluster_dec is not None) :
            ax.scatter(cluster_ra, cluster_dec,
                       transform=ax.get_transform('icrs'), color='cyan', marker='x', s=100,
                       label=cluster_name+' center')
            txt_clust = plt.text(cluster_ra, cluster_dec-0.2, cluster_name,
                             transform=ax.get_transform('fk5'), fontsize=10,
                             color='cyan', horizontalalignment='center',
                             verticalalignment='center')

        # Show the point sources
        for i in range(len(ps_name)): 
            if (ps_ra[i] is not None) * (ps_dec[i] is not None) :
                ax.scatter(ps_ra[i], ps_dec[i],
                           transform=ax.get_transform('icrs'), s=200, marker='o',
                           facecolors='none', edgecolors='green', label='Point sources')
                txt_ps = plt.text(ps_ra[i]-0.1, ps_dec[i]+0.1, ps_name[i],
                                  transform=ax.get_transform('fk5'),fontsize=10, color='green')
                
        # Show the PSF
        if PSF is not None:
            dec_mean_cor = np.cos((wcs_map.wcs.crval[1]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[1]+0.3)
                                  * np.pi/180.0)
            circle_ra = wcs_map.wcs.crval[0]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[0]/dec_mean_cor-0.3
            circle_dec = wcs_map.wcs.crval[1]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[1]+0.3
            circle_PSF = matplotlib.patches.Ellipse((circle_ra, circle_dec),
                                                    PSF/dec_mean_cor, PSF,
                                                    angle=0, linewidth=1, fill=True,
                                                    zorder=2, facecolor='lightgray',
                                                    edgecolor='white',
                                                    transform=ax.get_transform('fk5'))
            txt_ra  = wcs_map.wcs.crval[0]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[0]/dec_mean_cor-0.6
            txt_dec = wcs_map.wcs.crval[1]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[1]+0.3
            txt_psf = plt.text(txt_ra, txt_dec, 'PSF',
                               transform=ax.get_transform('fk5'), fontsize=12,
                               color='white',  verticalalignment='center')
            ax.add_patch(circle_PSF)

        # Formating and end
        ax.set_xlabel('R.A. (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title(maptitle)
        cbar = plt.colorbar()
        cbar.set_label(bartitle)
        #plt.legend(framealpha=1)
        fig.savefig(outfile)
        plt.close()

    else :
        print('!!!!!!!!!! WARNING: empty map, '+str(outfile)+' was not created')
        

#==================================================
# Show map
#==================================================

def show_profile(proffile, outfile,
                 expected_file=None,
                 theta500=None,
                 logscale=True):
    """
    Plot the profile to show.

    Parameters
    ----------
    Mandatory parameters:
    - mapfile (str): the map fits file to use
    - outfile (str): the ooutput plot file
    - cluster_t500 (deg): cluster theta 500

    Outputs
    --------
    - validation plot profile
    """

    set_default_plot_param()

    #---------- Read the data
    data = fits.open(proffile)[1]
    prof = data.data
    r_unit = data.columns['radius'].unit
    p_unit = data.columns['profile'].unit

    # Get the unit and adapt deg,arcmin,arcsec
    r_str_label = u.Unit(r_unit).to_string(format='latex_inline')
    p_str_label = u.Unit(p_unit).to_string(format='latex_inline')
    r_str_label = r_str_label.replace('{}^{\circ}', 'deg', 10)
    r_str_label = r_str_label.replace('{}^{\prime}', 'arcmin', 10)
    r_str_label = r_str_label.replace('{}^{\prime\prime}', 'arcsec', 10)
    p_str_label = p_str_label.replace('{}^{\circ}', 'deg', 10)
    p_str_label = p_str_label.replace('{}^{\prime}', 'arcmin', 10)
    p_str_label = p_str_label.replace('{}^{\prime\prime}', 'arcsec', 10)

    # Check if expected file is there
    if expected_file is not None:
        exp = fits.open(expected_file)[1]
        prof_exp = exp.data
        wnan_exp = np.isnan(prof_exp['profile']) * (prof_exp['radius'] > 0.5) # NaN at r>0.5 deg should be model=0
        prof_exp['profile'][wnan_exp] = 0.0 
        
    #---------- Plot
    w_pos = prof['profile'] >= 0
    w_neg = prof['profile'] < 0
    
    fig = plt.figure(1, figsize=(12, 8))

    # First frame
    if expected_file is not None:
        frame1 = fig.add_axes((.1,.3,.8,.6))
    else:
        frame1 = fig.add_axes((.1,.1,.8,.8))

    if logscale:
        plt.errorbar(prof['radius'][w_pos], prof['profile'][w_pos], yerr=prof['error'][w_pos],
                     color='blue', marker='o', linestyle='', label='data (> 0)')
        plt.errorbar(prof['radius'][w_neg], -prof['profile'][w_neg], yerr=prof['error'][w_neg],
                     color='cyan', marker='D', linestyle='', label='data (< 0)')
        xlim = [np.nanmin(prof['radius'])*0.5,         np.nanmax(prof['radius'])*1.1]
        ylim = [np.nanmin(prof['profile'][w_pos])*0.5, np.nanmax(prof['profile']+prof['error'])*1.5]
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.errorbar(prof['radius'], prof['profile'], yerr=prof['error'],
                     color='blue', marker='o', linestyle='', label='data')
        xlim = [0, np.nanmax(prof['radius'])*1.1]
        ylim = [np.nanmin(prof['profile'][w_pos]), np.nanmax(prof['profile']+prof['error'])]
        plt.xscale('linear')
        plt.yscale('linear')
        
    if expected_file is not None:
        plt.fill_between(prof_exp['radius'],
                         prof_exp['profile']+prof_exp['error'],
                         prof_exp['profile']-prof_exp['error'], color='red', alpha=0.2)
        plt.plot(prof_exp['radius'], prof_exp['profile'], color='red', label='Input model (IRF convolved)')
        plt.plot(prof_exp['radius'], prof_exp['profile']+prof_exp['error'],
                 color='red', linestyle='--', linewidth=0.5)
        plt.plot(prof_exp['radius'], prof_exp['profile']-prof_exp['error'],
                 color='red', linestyle='--', linewidth=0.5)
    
    if theta500 is not None:
        plt.vlines(theta500.to_value(r_unit), ymin=ylim[0], ymax=ylim[1],
                   color='orange', label='$\\theta_{500}$', linestyle='-.')

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel('Radius ('+r_str_label+')', color='k')
    plt.ylabel('Profile ('+p_str_label+')', color='k')
    if expected_file is not None: plt.xticks([])
    plt.legend()

    # Second frame
    if expected_file is not None:
        frame2 = fig.add_axes((.1,.1,.8,.2))

        itpl = interpolate.interp1d(prof_exp['radius'], prof_exp['profile'])
        prof_exp_itpl = itpl(prof['radius'])
        
        plt.plot(prof['radius'], (prof['profile']-prof_exp_itpl)/prof['error'],
                 color='k', marker='o', linestyle='')
        plt.hlines(0,  xlim[0], xlim[1], linestyle='-')
        plt.hlines(-3, xlim[0], xlim[1], linestyle='--')
        plt.hlines(+3, xlim[0], xlim[1], linestyle='--')

        if theta500 is not None:
            plt.vlines(theta500.to_value(r_unit), ymin=-5, ymax=5,
                       color='orange', label='$\\theta_{500}$', linestyle='-.')
    
        if logscale:
            plt.xscale('log')
        else:
            plt.xscale('linear')

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(-5, 5)
        plt.xlabel('Radius ('+r_str_label+')', color='k')
        plt.ylabel('$\\chi$')
    
    fig.savefig(outfile)
    plt.close()

    
#==================================================
# Quicklook of the event
#==================================================

def events_quicklook(evfile, outfile):
    """
    Basic plots directly made from the event file.

    Parameters
    ----------
    - evfile: event file name
    - outfile : output filename

    Outputs
    --------
    - event vizualisation plot
    """

    set_default_plot_param()
    
    events_hdu = fits.open(evfile)

    try:
        events_data1 = events_hdu[1].data
        events_data2 = events_hdu[2].data
        
        events_hdr0 = fits.getheader(evfile, 0)  # get default HDU (=0), i.e. primary HDU's header
        events_hdr1 = fits.getheader(evfile, 1)  # get primary HDU's header
        events_hdr2 = fits.getheader(evfile, 2)  # the second extension
        
        events_hdu.close()

        fig = plt.figure(1, figsize=(18, 14))
        
        # Plot the photon counts in RA-Dec
        Npt_plot = 1e5
        if len(events_data1) > Npt_plot:
            w = np.random.uniform(0,1,size=len(events_data1)) < Npt_plot/len(events_data1)
            events_data1_reduce = events_data1[w]
        else :
            events_data1_reduce = events_data1
        
        plt.subplot(221)
        plt.plot(events_data1_reduce['RA'], events_data1_reduce['DEC'], 'ko', ms=0.4, alpha=0.2)
        plt.xlim(np.amax(events_data1_reduce['RA']), np.amin(events_data1_reduce['RA']))
        plt.ylim(np.amin(events_data1_reduce['DEC']), np.amax(events_data1_reduce['DEC']))
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Photon coordinate map')
        plt.axis('scaled')
        
        # Energy histogram
        plt.subplot(222)
        plt.hist(np.log10(events_data1['ENERGY']), bins=50, color='black', log=True, alpha=0.3)
        plt.xlabel('log E/TeV')
        plt.ylabel('Photon counts')
        plt.title('Photon energy histogram')
        
        # Time counts histogram
        plt.subplot(223)
        plt.hist((events_data1['TIME'] - np.amin(events_data1['TIME']))/3600.0, bins=200,
                 log=False, color='black', alpha=0.3)
        plt.xlabel('Time (h)')
        plt.ylabel('Photon counts')
        plt.title('Photon time histogram')
        
        # Information
        plt.subplot(224)
        i1 = 'ObsID: '+events_hdr1['OBS_ID']
        i2 = 'Date obs: '+events_hdr1['DATE-OBS']+'-'+events_hdr1['TIME-OBS']
        i3 = 'Date end: '+events_hdr1['DATE-END']+'-'+events_hdr1['TIME-END']
        i4 = 'Live time: '+str(events_hdr1['LIVETIME'])+' '+events_hdr1['TIMEUNIT']
        t1 = 'Number of events: \n..... '+str(len(events_data1))
        t2 = 'Median energy: \n..... '+str(np.median(events_data1['ENERGY']))+events_hdr1['EUNIT']
        t3 = 'Median R.A.,Dec.: \n..... '+str(np.median(events_data1_reduce['RA']))+' deg \n..... '+str(np.median(events_data1_reduce['DEC']))+' deg'
        plt.text(0.1, 0.85, i1, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.80, i2, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.75, i3, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.70, i4, ha='left', rotation=0, wrap=True)

        plt.text(0.1, 0.55, t1, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.40, t2, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.20, t3, ha='left', rotation=0, wrap=True)
        plt.axis('off')
        
        fig.savefig(outfile, dpi=200)
        plt.close()
        
    except:
        print('')
        print('!!!!! Could not apply events_quicklook. Event file may be empty !!!!!')

        
#==================================================
# Get the pointing patern from a file
#==================================================

def get_pointings(filename):
    """
    Extract pointings from XML file

    Parameters
    ----------
    filename : str
        File name of observation definition XML file

    Returns
    -------
    pnt : list of dict
        Pointings
    """
    # Initialise pointings
    pnt = []

    # Open XML file
    xml = gammalib.GXml(filename)

    # Get observation list
    obs = xml.element('observation_list')

    # Get number of observations
    nobs = obs.elements('observation')

    # Loop over observations
    for i in range(nobs):

        # Get observation
        run = obs.element('observation', i)

        # Get pointing parameter
        npars   = run.elements('parameter')
        ra      = None
        dec     = None
        roi_ra  = None
        roi_dec = None
        roi_rad = None
        evfile  = None
        obsid   = run.attribute('id')
        for k in range(npars):
            par = run.element('parameter', k)
            if par.attribute('name') == 'Pointing':
                ra  = float(par.attribute('ra'))
                dec = float(par.attribute('dec'))
            elif par.attribute('name') == 'RegionOfInterest':
                roi_ra  = float(par.attribute('ra'))
                roi_dec = float(par.attribute('dec'))
                roi_rad = float(par.attribute('rad'))
            elif par.attribute('name') == 'EventList':
                evfile = par.attribute('file')

        # Add valid pointing
        if ra != None:
            p   = gammalib.GSkyDir()
            p.radec_deg(ra, dec)
            entry = {'l': p.l_deg(), 'b': p.b_deg(),
                     'ra': ra, 'dec': dec,
                     'roi_ra': roi_ra, 'roi_dec': roi_dec, 'roi_rad': roi_rad,
                     'evfile': evfile, 'obsid':obsid}
            pnt.append(entry)

    return pnt


#==================================================
# Plot the pointings
#==================================================

def show_pointings(xml_file, plotfile):
    """
    Plot information

    Parameters
    ----------
    - xml_file (str) : Observation definition xml file
    - plotfile (str): Plot filename
    """

    set_default_plot_param()

    pnt = get_pointings(xml_file)
    
    # Create figure
    plt.figure()
    fig = plt.figure(1, figsize=(10, 10))
    ax  = plt.subplot(111)
    colors = pl.cm.jet(np.linspace(0,1,len(pnt)))

    # Loop over pointings
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    i = 0
    for p in pnt:
        ra  = p['ra']
        dec = p['dec']
        roi_ra  = p['roi_ra']
        roi_dec = p['roi_dec']
        roi_rad = p['roi_rad']
        obsid   = p['obsid']

        #color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        color = colors[i]
        
        ax.scatter(ra, dec, s=150, marker='x', color=color)
        circle = matplotlib.patches.Ellipse(xy=(roi_ra, roi_dec),
                                            width=2*roi_rad/np.cos(dec*np.pi/180),
                                            height=2*roi_rad,
                                            alpha=0.1,
                                            linewidth=1,
                                            #color=color,
                                            facecolor=color,
                                            edgecolor=color, label='ObsID'+obsid)
        ax.add_patch(circle)
        
        xmin.append(roi_ra-roi_rad/np.cos(dec*np.pi/180))
        xmax.append(roi_ra+roi_rad/np.cos(dec*np.pi/180))
        ymin.append(roi_dec-roi_rad)
        ymax.append(roi_dec+roi_rad)
        i += 1

    xctr = (np.amax(xmax) + np.amin(xmin)) / 2.0
    yctr = (np.amax(ymax) + np.amin(ymin)) / 2.0
    fovx = (np.amax(xmax) - np.amin(xmin))*1.1/np.cos(yctr*np.pi/180)
    fovy = (np.amax(ymax) - np.amin(ymin))*1.1
        
    plt.xlim(xctr+fovx/2, xctr-fovx/2)
    plt.ylim(yctr-fovy/2, yctr+fovy/2)
    plt.legend()
        
    # Plot title and labels
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec. (deg)')

    # Show plots or save it into file
    plt.savefig(plotfile)
    plt.close()

    return


#==================================================
# Plot the pointings
#==================================================

def show_obsdef(xml_file, coord, plotfile):
    """
    Plot information

    Parameters
    ----------
    - xml_file (str) : Observation definition xml file
    - coord (SkyCoord): coordinates of the target
    - plotfile (str): Plot filename
    """

    set_default_plot_param()
    
    info = plotting_obsfile.run_csobsinfo(xml_file,
                                          coord.icrs.ra.to_value('deg'),
                                          coord.icrs.dec.to_value('deg'))
    plotting_obsfile.plot_information(info,
                                      coord.icrs.ra.to_value('deg'),
                                      coord.icrs.dec.to_value('deg'),
                                      plotfile)
    
    return


#==================================================
# Plot the spectrum of the sources in a model
#==================================================

def show_model_spectrum(xml_file, plotfile,
                        emin=0.01, emax=100.0, enumbins=100):
    """
    Plot information

    Parameters
    ----------
    - xml_file (str) : Observation definition xml file
    - plotfile (str): Plot filename
    - emin (min energy): minimal energy in TeV
    - emax (max energy): maximal energy in TeV
    - enumbins (int): number of energy bins

    """

    set_default_plot_param()

    # Setup energy axis
    e_min   = gammalib.GEnergy(emin, 'TeV')
    e_max   = gammalib.GEnergy(emax, 'TeV')
    ebounds = gammalib.GEbounds(enumbins, e_min, e_max)

    # Read models XML file
    models = gammalib.GModels(xml_file)

    # Plot spectra in loop
    plt.figure(figsize=(12,8))
    plt.loglog()
    plt.grid()
    colors = pl.cm.jet(np.linspace(0,1,len(models)))
    
    for imod in range(len(models)):
        model = models[imod]
        if model.type() == 'DiffuseSource' or model.type() == 'PointSource':
            spectrum = model.spectral()
            # Setup lists of x and y values
            x   = []
            y   = []
            for i in range(enumbins):
                energy = ebounds.elogmean(i)
                value  = spectrum.eval(energy)
                x.append(energy.TeV())
                y.append(value)
            plt.loglog(x, y, linewidth=3, color=colors[imod], label=model.name()+' ('+spectrum.type()+')')

    plt.xlabel('Energy (TeV)')
    plt.ylabel(r'dN/dE (ph s$^{-1}$ cm$^{-2}$ MeV$^{-1}$)')
    plt.legend()
    plt.savefig(plotfile)
    plt.close()
    
    return


#==================================================
# Plot the spectrum of analysed sources
#==================================================

def show_spectrum(specfile, outfile, butfile=None, expected_file=None):
    """
    Plot the spectrum to show.

    Parameters
    ----------
    - mapfile (str): the spectrum fits file to use
    - outfile (str): the output plot file
    - butfile (str): the buterfly file

    Outputs
    --------
    - validation plot spectrum
    """

    set_default_plot_param()
    
    #----- Read the data
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
        
    # Buterfly plot can be added
    if butfile is not None:
        f = open(butfile, "r")
        lines = f.readlines()
        b_energy = []
        for x in lines: b_energy.append(x.split(' ')[0])
        b_energy = np.array(b_energy).astype(np.float)*u.MeV
        b_bf= []
        for x in lines: b_bf.append(x.split(' ')[1])
        b_bf = np.array(b_bf).astype(np.float) / u.cm**2 / u.MeV / u.s
        b_ll= []
        for x in lines: b_ll.append(x.split(' ')[2])
        b_ll = np.array(b_ll).astype(np.float) / u.cm**2 / u.MeV / u.s
        b_ul = []
        for x in lines: b_ul.append(x.split(' ')[3])
        b_ul = np.array(b_ul).astype(np.float) / u.cm**2 / u.MeV / u.s
        f.close()
 
    # Expected data plot can be added
    if expected_file is not None:
        f = open(expected_file, "r")
        lines = f.readlines()
        E_exp = []
        for x in lines: E_exp.append(x.split('                          ')[0])
        E_exp = np.array(E_exp[1:]).astype(np.float)*u.MeV
        dNdEdSdt_exp = []
        for x in lines: dNdEdSdt_exp.append(x.split('                          ')[1])
        dNdEdSdt_exp = np.array(dNdEdSdt_exp[1:]).astype(np.float)*u.MeV**-1*u.s**-1*u.cm**-2
        f.close()

        max_expected = np.nanmax((E_exp**2 * dNdEdSdt_exp).to_value('GeV cm-2 s-1'))*u.GeV/u.cm**2/u.s

        itpl = interpolate.interp1d(E_exp.to_value('GeV'), (E_exp**2*dNdEdSdt_exp).to_value('GeV cm-2 s-1'))
        dNdEdSdt_exp_itpl = itpl(energy.to_value('GeV'))*u.GeV/u.cm**2/u.s

    #----- Define "good" and "bad" points
    TSg = TS*1
    TSg[TSg <= 1] = 1
    w0 = (e_flux==0)
    e_flux[w0] = -1*u.erg/u.cm**2/u.s
    wgood  = (flux/e_flux > 2) * (e_flux > 0) * (np.sqrt(TSg) > 2)
    e_flux[w0] = 0*u.erg/u.cm**2/u.s
    wbad   = ~wgood

    #----- Define the range
    rngyp = 1.2*np.nanmax((flux+e_flux).to_value('GeV cm-2 s-1'))*u.GeV/u.cm**2/u.s
    if expected_file is not None:
        if max_expected*1.2 > rngyp:
            rngyp = max_expected*1.2
    if np.sum(wgood) > 0 :
        rngym = 0.5*np.nanmin(flux[wgood].to_value('GeV cm-2 s-1'))*u.GeV/u.cm**2/u.s
    else :
        rngym = 1e-14*u.erg/u.cm**2/u.s # CTA sensitivity         

    rngxm = np.nanmin(energy.to_value('GeV')-ed_Energy.to_value('GeV'))*u.GeV*0.5
    rngxp = np.nanmax(energy.to_value('GeV')+eu_Energy.to_value('GeV'))*u.GeV*1.5
    if expected_file is not None:
        rngxm = np.nanmin([np.nanmin(E_exp.to_value('GeV')), rngxm.to_value('GeV')])*u.GeV
        rngxp = np.nanmax([np.nanmax(E_exp.to_value('GeV')), rngxp.to_value('GeV')])*u.GeV
        
    xlim = [rngxm, rngxp]
    ylim = [rngym, rngyp]
        
    #----- Start the plot
    fig = plt.figure(figsize=(12,8))
    if expected_file is not None:
        gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1])
    else:
        gs = GridSpec(1,1)
        ax1 = plt.subplot(gs[0])

    if butfile is not None:
        ax1.plot(b_energy.to_value('GeV'), (b_energy**2 * b_bf).to_value('GeV cm-2 s-1'),
                 ls='-', color='blue', label='68% CL')
        ax1.plot(b_energy.to_value('GeV'), (b_energy**2 * b_ul).to_value('GeV cm-2 s-1'),
                 ls='-.', color='blue')
        ax1.plot(b_energy.to_value('GeV'), (b_energy**2 * b_ll).to_value('GeV cm-2 s-1'),
                 ls='-.', color='blue')
        ax1.fill_between(b_energy.to_value('GeV'), (b_energy**2 * b_ll).to_value('GeV cm-2 s-1'),
                         (b_energy**2 * b_ul).to_value('GeV cm-2 s-1'),
                         color='blue', alpha=0.2)

    if expected_file is not None:
        ax1.plot(E_exp.to_value('GeV'), (E_exp**2 * dNdEdSdt_exp).to_value('GeV cm-2 s-1'),
                 ls='--', linewidth=2, color='k', label='Expected spectrum')

    # Measured spectrum
    ax1.errorbar(energy[wgood].to_value('GeV'), flux[wgood].to_value('GeV cm-2 s-1'),
                 yerr=e_flux[wgood].to_value('GeV cm-2 s-1'),
                 xerr=[ed_Energy[wgood].to_value('GeV'), eu_Energy[wgood].to_value('GeV')],
                 marker='o', elinewidth=2, color='red',
                 markeredgecolor="black", markerfacecolor="red",
                 ls ='', label='Data')
    ax1.errorbar(energy[wbad].to_value('GeV'), UpperLimit[wbad].to_value('GeV cm-2 s-1'),
                 xerr=[ed_Energy[wbad].to_value('GeV'), eu_Energy[wbad].to_value('GeV')],
                 yerr=0.1*UpperLimit[wbad].to_value('GeV cm-2 s-1'), uplims=True,
                 marker="", elinewidth=2, color="pink",
                 markeredgecolor="pink", markerfacecolor="pink",
                 linestyle="None")
    ax1.set_ylabel('$E^2 \\frac{dN}{dEdSdt}$ (GeV/cm$^2$/s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xlim[0].to_value('GeV'),          xlim[1].to_value('GeV'))
    ax1.set_ylim(ylim[0].to_value('GeV cm-2 s-1'), ylim[1].to_value('GeV cm-2 s-1'))
    if expected_file is not None:
        ax1.set_xticks([])
    else:
        ax1.set_xlabel('Energy (GeV)')
    ax1.legend()
    
    # Add extra unit axes
    ax2 = ax1.twinx()
    ax2.plot(energy[wgood].to_value('GeV'), flux[wgood].to_value('erg cm-2 s-1'), 'k-', alpha=0.0)
    ax2.set_ylabel('$E^2 \\frac{dN}{dEdSdt}$ (erg/cm$^2$/s)')
    ax2.set_yscale('log')
    ax2.set_ylim(ylim[0].to_value('erg cm-2 s-1'), ylim[1].to_value('erg cm-2 s-1'))

    # Residual plot
    if expected_file is not None:
        wplt = (e_flux != 0)
        e_flux[~wplt] = -1*u.erg/u.cm**2/u.s
        w_gt5 = wplt * (((flux-dNdEdSdt_exp_itpl).to_value('GeV cm-2 s-1'))/e_flux.to_value('GeV cm-2 s-1') > 5)
        w_lt5 = wplt * (((flux-dNdEdSdt_exp_itpl).to_value('GeV cm-2 s-1'))/e_flux.to_value('GeV cm-2 s-1') < -5)
        e_flux[~wplt] = 0*u.erg/u.cm**2/u.s
        ax3.plot(energy[wplt].to_value('GeV'),
                 ((flux[wplt]-dNdEdSdt_exp_itpl[wplt]).to_value('GeV cm-2 s-1'))/e_flux[wplt].to_value('GeV cm-2 s-1'),
                 linestyle='', marker='o', color='k')

        ax3.errorbar(energy[w_gt5].to_value('GeV'), energy[w_gt5].to_value('GeV')*0+4,
                     yerr=0.5, lolims=True,
                     marker="", elinewidth=2, color="pink",
                     markeredgecolor="pink", markerfacecolor="pink",
                     linestyle="None")

        ax3.errorbar(energy[w_lt5].to_value('GeV'), energy[w_lt5].to_value('GeV')*0-4,
                     yerr=0.5, uplims=True,
                     marker="", elinewidth=2, color="pink",
                     markeredgecolor="pink", markerfacecolor="pink",
                     linestyle="None")
        
        ax3.axhline(0,  xlim[0].to_value('GeV')*0.1, xlim[1].to_value('GeV')*10, linestyle='-', color='k')
        ax3.axhline(-3, xlim[0].to_value('GeV')*0.1, xlim[1].to_value('GeV')*10, linestyle='--', color='k')
        ax3.axhline(+3, xlim[0].to_value('GeV')*0.1, xlim[1].to_value('GeV')*10, linestyle='--', color='k')
        ax3.set_xlabel('Energy (GeV)')
        ax3.set_ylabel('$\\chi$')
        ax3.set_xscale('log')
        ax3.set_xlim(xlim[0].to_value('GeV'), xlim[1].to_value('GeV'))
        ax3.set_ylim(-5, 5)
        
    fig.savefig(outfile)
    plt.close()

#==================================================
# Plot the residual spectrum of analysed sources
#==================================================

def show_spectrum_residual(specfile, outfile):
    """
    Plot the spectrum residual.

    Parameters
    ----------
    Mandatory parameters:
    - specfile (str): the spectrum fits file to use
    - outfile (str): the output plot file

    Outputs
    --------
    - validation plot spectrum
    """

    set_default_plot_param()
    
    #----- Read the data
    hdu = fits.open(specfile)
    spectrum = hdu[1].data
    hdu.close()
    
    emin       = spectrum['Emin']*u.TeV
    emax       = spectrum['Emax']*u.TeV
    emean      = np.sqrt(emin * emax)
    counts     = spectrum['Counts']
    model      = spectrum['Model']
    resid      = spectrum['Residuals']

    estep = [emin[0].to_value('GeV')]
    for i in range(len(emax)): estep.append(emax[i].to_value('GeV'))
    estep = estep*u.GeV
    
    #----- Plot the data
    fig = plt.figure(1, figsize=(12, 8))
    frame1 = fig.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(emean.to_value('GeV'), counts, yerr=np.sqrt(counts), xerr=[(emean-emin).to_value('GeV'),
                                                                            (emax-emean).to_value('GeV')],
                 fmt='ko', capsize=0, linewidth=2, zorder=2, label='Data')
    plt.step(estep.to_value('GeV'), np.append(model, model[-1]), where='post', color='0.5',
             linewidth=2, label='Model')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.yscale('log')

    skiplist = ['Counts', 'Model', 'Residuals', 'Counts_Off', 'Model_Off', 'Residuals_Off', 'Emin', 'Emax']
    for s in range(len(spectrum.columns)):
        if spectrum.columns[s].name not in skiplist:
            component = spectrum[spectrum.columns[s].name]
            plt.step(estep.to_value('GeV'), np.append(component, component[-1]), where='post',
                     label=spectrum.columns[s].name)

    plt.legend(loc='best')

    frame2 = fig.add_axes((.1,.1,.8,.2))        
    plt.errorbar(emean.to_value('GeV'), resid, yerr=1.0, xerr=[(emean-emin).to_value('GeV'),
                                                               (emax-emean).to_value('GeV')],
                      fmt='ko', capsize=0, linewidth=2, zorder=2)
    plt.axhline(0, color='0.5', linestyle='--')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Residual ($\sigma$)')
    plt.xscale('log')
            
    fig.savefig(outfile)
    plt.close()

    
#==================================================
# Plot the residual spectrum of analysed sources
#==================================================

def show_param_cormat(covfile, outfile):
    """
    Plot the parameter covariance matrix.

    Parameters
    ----------
    Mandatory parameters:
    - covfile (str): the covariance matrix fits file
    - outfile (str): the output plot file

    Outputs
    --------
    - validation plot
    """

    set_default_plot_param(leftspace=0.28, rightspace=0.77)

    #----- Read the data
    hdul = fits.open(covfile)
    dat = hdul[1].data
    par = dat['Parameters'][0].split()
    cov = dat['Covariance'][0,:,:]

    #----- Get the fixed param out
    diag = np.diag(cov)
    idx = np.where(diag == 0)[0]
    wkeep = np.where(diag != 0)[0]
    if len(idx) != 0:
        cov = np.delete(cov, idx, 0)
        cov = np.delete(cov, idx, 1)
        par = np.array(par)[wkeep]

    #----- Compute the correlation matrix
    cor = correlation_from_covariance(cov)

    #----- show the plot
    fig = plt.figure(1, figsize=(15, 10))
    im, cbar = heatmap(cor, par, par, vmin=-1, vmax=1,
                       cmap='RdBu', cbarlabel="Correlation matrix", origin='lower')
    fig.savefig(outfile)
    plt.close()

