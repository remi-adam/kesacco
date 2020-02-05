"""
This file contains the CTAsim class. It is dedicated to the construction of a 
CTAsim object, which defines how CTA observations of a cluster would proceed.
These simulations of observations are then performed and a quicklook analysis 
is available.

"""

#==================================================
# Requested imports
#==================================================

import os
import copy
import numpy as np
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

from ClusterPipe.Tools import make_cluster_template
from ClusterPipe       import clustpipe_sim_run
from ClusterPipe       import clustpipe_sim_plot


#==================================================
# Cluster class
#==================================================

class CTAsim(object):
    """ 
    CTAsim class. 
    This class serves as parser to the ClusterPipe class and 
    contains simulation related tools.
    
    Methods
    ----------  
    - run_sim_obs(self, obsID=None): run the observations to generate event files
    - run_sim_quicklook(self, obsID=None): perform quicklook analysis of event files and model
    
    """

    #==================================================
    # Run the observations
    #==================================================
    
    def run_sim_obs(self, obsID=None):
        """
        Run the observations
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be observed
        
        """

        #----- Create the output directory if needed
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        if not os.path.exists(self.output_dir+'/ClusterModel'):
            os.mkdir(self.output_dir+'/ClusterModel')
            
        #----- Get the obs ID to run (defaults is all of them)
        obsID = self._check_obsID(obsID)
            
        if not self.silent:
            print('----- ObsID to be observed:')
            print(obsID)

        #----- Build the cluster template
        # Make sure the cluster FoV matches all requested observations
        self.match_cluster_to_pointing()
        
        # Warning regarding pixel size
        if (not self.silent) and (self.cluster.map_reso > 0.02*u.deg):
            print('WARNING: the FITS map resolution is larger than 0.02 deg, i.e. the PSF at 100 TeV')
        
        # Make cluster templates        
        make_cluster_template.make_map(self.cluster,
                                       self.output_dir+'/ClusterModel/SimuMap.fits',
                                       Egmin=self.obs_setup.get_emin(),
                                       Egmax=self.obs_setup.get_emax(),
                                       includeIC=True)
        
        make_cluster_template.make_spectrum(self.cluster,
                                            self.output_dir+'/ClusterModel/SimuSpectrum.txt',
                                            energy=np.logspace(-1,5,1000)*u.GeV,
                                            includeIC=True)
        
        #----- Run the observations
        Nobs_done = 0
        for iobs in obsID:
            
            # Select the setup of the corresponding run
            setup = self.obs_setup.select_obs(iobs)
            
            # Define the subdirectory for the run
            output_dir = self.output_dir+'/ObsID'+iobs
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            
            # Run the simulation
            clustpipe_sim_run.run(output_dir,
                                  self.output_dir+'/ClusterModel/SimuMap.fits',
                                  self.output_dir+'/ClusterModel/SimuSpectrum.txt',
                                  self.cluster,
                                  self.compact_source,
                                  setup,
                                  silent=self.silent)
            
            # Information
            Nobs_done += 1
            if not self.silent: print('----- Observation '+str(Nobs_done)+'/'+str(len(obsID))+' done')

        
    #==================================================
    # Quicklook
    #==================================================

    def run_sim_quicklook(self, obsID=None,
                          ClusterModelOnly=False,
                          EventOnly=False,
                          bkgsubtract=None,
                          smoothing_FWHM=0.03*u.deg):
        """
        Provide quicklook analysis of the simulation
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be observed
        - ClusterModelOnly (bool): set to true to show only the model
        - EventOnly (bool): set to true to show only the event file 
        related quicklook
        - bkgsubtract (bool): apply IRF background subtraction in skymap
        - smoothing_FWHM (quantity): apply smoothing to skymap
        
        """
        
        #----- Get the obs ID to run (defaults is all of them)
        obsID = self._check_obsID(obsID)
        
        if not self.silent: print('----- ObsID to be quicklooked:')
        if not self.silent: print(obsID)
        
        #----- Show the cluster model
        if not EventOnly:
            self.match_cluster_to_pointing()
            self.cluster.output_dir = self.output_dir+'/ClusterModel'
            self.cluster.plot()
            
        #----- Run the quicklook for eventfiles
        if not ClusterModelOnly:
            Nobs_done = 0
            for iobs in obsID:
                
                # Define the subdirectory for the run
                output_dir = self.output_dir+'/ObsID'+iobs
                
                # Run the simulation
                clustpipe_sim_plot.main(output_dir,
                                        self.cluster,
                                        self.compact_source,
                                        self.obs_setup.select_obs(iobs),
                                        map_reso=self.cluster.map_reso,
                                        bkgsubtract=bkgsubtract,
                                        smoothing_FWHM=smoothing_FWHM,
                                        silent=self.silent)
                
                # Information
                Nobs_done += 1
                if not self.silent: print('----- Quicklook '+str(Nobs_done)+'/'+str(len(obsID))+' done')
            
            
