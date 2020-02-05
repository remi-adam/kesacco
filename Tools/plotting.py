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
from matplotlib.colors import SymLogNorm
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy import interpolate
import scipy.ndimage as ndimage

#==================================================
# Style
#==================================================

cta_energy_range   = [0.02, 100.0]*u.TeV
fermi_energy_range = [0.1, 300.0]*u.GeV

def set_default_plot_param():
    
    dict_base = {'font.size':        16, 
                 'legend.fontsize':  16,
                 'xtick.labelsize':  16,
                 'ytick.labelsize':  16,
                 'axes.labelsize':   16,
                 'axes.titlesize':   16,
                 'figure.titlesize': 16,
                 'figure.figsize':[8.0, 6.0],
                 'figure.subplot.right':0.97,
                 'figure.subplot.left':0.18, # Ensure enough space on the left so that all plot can be aligned
                 'font.family':'serif',
                 'figure.facecolor': 'white',
                 'legend.frameon': True}

    plt.rcParams.update(dict_base)

    
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
        norm = []
        for i in range(100):
            norm.append(np.std(ndimage.gaussian_filter(np.random.normal(loc=0.0, scale=1.0, size=image.shape), sigma=sigma_sm)))
        image = image / np.mean(norm)
        print('WARNING: the significance is boosted accounting for smoothing assuming gaussian noise')
        
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
        fig = plt.figure(1, figsize=(12, 12))
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
                                                    transform=ax.get_transform('fk5'))
            ax.add_patch(circle_500)
            txt_r500 = plt.text(cluster_ra - cluster_t500, cluster_dec - cluster_t500,
                                '$R_{500}$',
                                transform=ax.get_transform('fk5'), fontsize=10,
                                color='lightgray',
                                horizontalalignment='center',verticalalignment='center')

        # Show the pointing
        if (ptg_ra is not None) * (ptg_dec is not None) :
            ax.scatter(ptg_ra, ptg_dec,
                       transform=ax.get_transform('icrs'), color='white', marker='x', s=100)
            txt_ptg = plt.text(ptg_ra, ptg_dec+0.2, 'Pointing',
                               transform=ax.get_transform('fk5'),fontsize=10,
                               color='white', horizontalalignment='center',
                               verticalalignment='center')
        # Show the cluster center
        if (cluster_ra is not None) * (cluster_dec is not None) :
            ax.scatter(cluster_ra, cluster_dec,
                       transform=ax.get_transform('icrs'), color='cyan', marker='x', s=100)
            txt_clust = plt.text(cluster_ra, cluster_dec-0.2, cluster_name,
                             transform=ax.get_transform('fk5'), fontsize=10,
                             color='cyan', horizontalalignment='center',
                             verticalalignment='center')

        # Show the point sources
        for i in range(len(ps_name)): 
            if (ps_ra[i] is not None) * (ps_dec[i] is not None) :
                ax.scatter(ps_ra[i], ps_dec[i],
                           transform=ax.get_transform('icrs'), s=200, marker='o',
                           facecolors='none', edgecolors='green')
                txt_ps = plt.text(ps_ra[i]-0.1, ps_dec[i]+0.1, ps_name[i],
                                  transform=ax.get_transform('fk5'),fontsize=10, color='green')
                
        # Show the PSF
        if not None:
            dec_mean_cor = np.cos((wcs_map.wcs.crval[1]-(wcs_map.wcs.crpix*wcs_map.wcs.cdelt)[1]+0.3) * np.pi/180.0)
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
        fig.savefig(outfile, bbox_inches='tight')
        plt.close()

    else :
        print('!!!!!!!!!! WARNING: empty map, '+str(outfile)+' was not created')

        
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
        plt.hist((events_data1['TIME'] - np.amin(events_data1['TIME']))/3600.0, bins=200, log=False, color='black', alpha=0.3)
        plt.xlabel('Time (h)')
        plt.ylabel('Photon counts')
        plt.title('Photon time histogram')
        
        # Information
        plt.subplot(224)
        i0 = 'ObsID: '+events_hdr1['OBS_ID']
        i1 = 'Target: '+events_hdr1['OBJECT']
        i2 = 'Date obs: '+events_hdr1['DATE-OBS']+'-'+events_hdr1['TIME-OBS']
        i3 = 'Date end: '+events_hdr1['DATE-END']+'-'+events_hdr1['TIME-END']
        i4 = 'Live time: '+str(events_hdr1['LIVETIME'])+' '+events_hdr1['TIMEUNIT']
        t1 = 'Number of events: \n..... '+str(len(events_data1))
        t2 = 'Median energy: \n..... '+str(np.median(events_data1['ENERGY']))+events_hdr1['EUNIT']
        t3 = 'Median R.A.,Dec.: \n..... '+str(np.median(events_data1_reduce['RA']))+' deg \n..... '+str(np.median(events_data1_reduce['DEC']))+' deg'
        plt.text(0.1, 0.90, i0, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.85, i1, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.80, i2, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.75, i3, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.70, i4, ha='left', rotation=0, wrap=True)

        plt.text(0.1, 0.55, t1, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.40, t2, ha='left', rotation=0, wrap=True)
        plt.text(0.1, 0.20, t3, ha='left', rotation=0, wrap=True)
        plt.axis('off')
        
        fig.savefig(outfile, bbox_inches='tight', dpi=200)
        plt.close()
        
    except:
        print('')
        print('!!!!! Could not apply events_quicklook. Event file may be empty !!!!!')
        
