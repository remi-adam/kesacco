"""
This file contains plotting tools for the CTA simulation.

"""

#==================================================
# Requested imports
#==================================================

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.io import fits

from ClusterPipe.Tools import tools_imaging
from ClusterPipe.Tools import plotting
from ClusterPipe.Tools import utilities


#==================================================
# Quicklook of the skymap
#==================================================

def skymap_quicklook(output_file,
                     evfile,
                     setup_obs,
                     compact_source,
                     cluster,
                     map_reso=0.01*u.deg,
                     smoothing_FWHM=0.0*u.deg,
                     bkgsubtract='NONE',
                     silent=True,
                     MapCenteredOnTarget=True):
    """
    Sky maps to show the data.
    
    Parameters
    ----------
    - output_dir (str): output directory
    - evfile (str): eventfile (full name)
    - setup_obs (ObsSetup object): object that contains
    the properties of the observing setup
    - compact_source (CompactSource object): object that contains
    the properties of the compact_sources
    - cluster (ClusterModel): object used for the cluster model
    - map_reso (quantity): skymap resolution, homogeneous to deg
    - smoothing_FWHM (quantity): apply smoothing to skymap
    - bkgsubtract (bool): apply IRF background subtraction in skymap
    - MapCenteredOnTarget (bool): center the map on the cluster (or pointing)

    Outputs
    --------
    - validation plot map
    """
    
    #---------- Get the number of pixels
    npix = utilities.npix_from_fov_def(setup_obs.rad[0], map_reso)
    
    #---------- Get the point sources
    ps_name  = []
    ps_ra    = []
    ps_dec   = []
    for i in range(len(compact_source.name)):
        ps_name.append(compact_source.name[i])
        ps_ra.append(compact_source.spatial[i]['param']['RA']['value'].to_value('deg'))
        ps_dec.append(compact_source.spatial[i]['param']['DEC']['value'].to_value('deg'))
    
    #---------- Get the PSF at the considered energy
    CTA_PSF = plotting.get_cta_psf(setup_obs.caldb[0], setup_obs.irf[0],
                                   setup_obs.emin[0].to_value('TeV'), setup_obs.emax[0].to_value('TeV'))
    PSF_tot = np.sqrt(CTA_PSF**2 + smoothing_FWHM.to_value('deg')**2)

    #---------- Choose map center
    if MapCenteredOnTarget:
        cntr_ra  = cluster.coord.icrs.ra.to_value('deg')
        cntr_dec = cluster.coord.icrs.dec.to_value('deg')
    else:
        cntr_ra  = setup_obs.coord[0].icrs.ra.to_value('deg')
        cntr_dec = setup_obs.coord[0].icrs.dec.to_value('deg') 
    
    #---------- Compute skymap
    skymap = tools_imaging.skymap(evfile, output_file+'.fits',
                                  npix, map_reso.to_value('deg'),
                                  cntr_ra, cntr_dec,
                                  emin=setup_obs.emin[0].to_value('TeV'),
                                  emax=setup_obs.emax[0].to_value('TeV'),
                                  caldb=setup_obs.caldb[0], irf=setup_obs.irf[0],
                                  bkgsubtract=bkgsubtract)
    if silent == False:
        print('')
        print(skymap)
        print('')
    
    #---------- Plot
    plotting.show_map(output_file+'.fits',
                      output_file+'.pdf',
                      smoothing_FWHM=smoothing_FWHM,
                      cluster_ra=cluster.coord.icrs.ra.to_value('deg'),
                      cluster_dec=cluster.coord.icrs.dec.to_value('deg'),
                      cluster_t500=cluster.theta500.to_value('deg'),
                      cluster_name=cluster.name,
                      ps_name=ps_name,
                      ps_ra=ps_ra,
                      ps_dec=ps_dec,
                      ptg_ra=setup_obs.coord[0].icrs.ra.to_value('deg'),
                      ptg_dec=setup_obs.coord[0].icrs.dec.to_value('deg'),
                      PSF=PSF_tot,
                      bartitle='Counts',
                      rangevalue=[None, None],
                      logscale=True,
                      significance=False,
                      cmap='magma')
    
    
#==================================================
# Main function
#==================================================

def main(output_dir,
         cluster,
         compact_source,
         setup_obs,
         map_reso=0.01*u.deg,
         bkgsubtract='NONE',
         smoothing_FWHM=0.03*u.deg,
         silent=False,
         MapCenteredOnTarget=True):
    """
    Script of the simulation pipeline dedicated to plot validations of the results.

    Parameters
    ----------
    - output_dir: dictionary where to look fore products
    - cluster (ClusterModel): object used for the cluster model
    - compact_source (CompactSource object): object that contains
    the properties of the compact_sources
    - setup_obs (ObsSetup object): object that contains
    the properties of the observing setup
    - map_reso (quantity): skymap resolution, homogeneous to deg
    - bkgsubtract (bool): apply IRF background subtraction in skymap
    - smoothing_FWHM (quantity): apply smoothing to skymap
    - silent (bool): print info or not
    - MapCenteredOnTarget (bool): to center the skymaps on target 
    or pointing

    Outputs
    --------
    - validation plots
    """
    
    #---------- Plot the events
    plotting.events_quicklook(output_dir+'/Events'+setup_obs.obsid[0]+'.fits', output_dir+'/Events'+setup_obs.obsid[0]+'.png')
    
    #---------- Skymaps    
    skymap_quicklook(output_dir+'/Sim_Skymap'+setup_obs.obsid[0],
                     output_dir+'/Events'+setup_obs.obsid[0]+'.fits',
                     setup_obs, compact_source, cluster,
                     map_reso=map_reso, smoothing_FWHM=smoothing_FWHM, bkgsubtract=bkgsubtract,
                     silent=silent, MapCenteredOnTarget=MapCenteredOnTarget)
