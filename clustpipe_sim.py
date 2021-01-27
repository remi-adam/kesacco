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
import astropy.units as u
from random import randint
import gammalib
import ctools

from kesacco.Tools import plotting


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
    # Run observation simulation
    #==================================================
    
    def run_sim_obs(self, obsID=None, seed=None):
        """
        Run all the observations at once
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be observed
        - seed (int): the seed used for simulations of observations
        """

        #----- Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        #----- Check onoff/Edisp
        if not not self.silent and self.method_ana == 'ONOFF' and self.spec_edisp == False:
            print('-----------------------------------------------------------------------------')
            print('WARNING: The events are generated without accounting for energy dispersion.  ')
            print('         The current analysis method is set to ONOFF, which necessarily      ')
            print('         accounts for energy dispersion and might thus leads to biases later.')
            print('-----------------------------------------------------------------------------')

        #----- Get the obs ID to run
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be observed: '+str(obsID))
        self.obs_setup.match_bkg_id() # make sure Bkg are unique

        #----- Make sure the cluster FoV matches all requested observations
        self._match_cluster_to_pointing()
        
        #----- Make cluster templates
        model_sim = self._make_model(prefix='Sim_Model', includeIC=False)
        self.cluster.save_param()
        os.rename(self.cluster.output_dir+'/parameters.txt',
                  self.cluster.output_dir+'/Sim_Model_Cluster_param.txt')
        os.rename(self.cluster.output_dir+'/parameters.pkl',
                  self.cluster.output_dir+'/Sim_Model_Cluster_param.pkl')

        #----- Make observation files
        self.obs_setup.write_pnt(self.output_dir+'/Sim_Pnt.def', obsid=obsID)
        self.obs_setup.run_csobsdef(self.output_dir+'/Sim_Pnt.def', self.output_dir+'/Sim_ObsDef.xml')

        #----- Get the seed for reapeatable simu
        if seed is None: seed = randint(1, 1e6)
        
        #----- Run the observation
        obssim = ctools.ctobssim()
        obssim['inobs']      = self.output_dir+'/Sim_ObsDef.xml'
        obssim['inmodel']    = self.output_dir+'/Sim_Model_Unstack.xml'
        #obssim['caldb']     = --> read from ObsDef
        #obssim['irf']       = --> read from ObsDef
        obssim['edisp']      = self.spec_edisp
        obssim['outevents']  = self.output_dir+'/Events.xml'
        obssim['prefix']     = self.output_dir+'/TmpEvents'
        obssim['startindex'] = 1
        obssim['seed']       = seed
        #obssim['ra']        = --> read from ObsDef
        #obssim['dec']       = --> read from ObsDef
        #obssim['rad']       = --> read from ObsDef
        #obssim['tmin']      = --> read from ObsDef
        #obssim['tmax']      = --> read from ObsDef
        obssim['mjdref']     = self.time_mjdref
        #obssim['emin']      = --> read from ObsDef
        #obssim['emax']      = --> read from ObsDef
        #obssim['deadc']     = --> read from ObsDef
        obssim['maxrate']    = 1e6
        obssim['logfile']    = self.output_dir+'/Events_log.txt'
        obssim['chatter']    = 2
        obssim.logFileOpen()
        obssim.execute()
        obssim.logFileClose()
        
        if not self.silent:
            print('------- Simulation log -------')
            print(obssim)
            print('')

        self._correct_eventfile_names(self.output_dir+'/Events.xml', prefix='Events')
        
        
    #==================================================
    # Quicklook
    #==================================================

    def run_sim_quicklook(self,
                          obsID=None,
                          ShowObsDef=True,
                          ShowSkyModel=True,
                          ShowEvent=True,
                          bkgsubtract='NONE',
                          smoothing_FWHM=0.1*u.deg,
                          MapCenteredOnTarget=True):
        """
        Provide quicklook analysis of the simulation
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be quicklooked
        - ShowObsDef (bool): set to true to show the observation definition
        - ShowEvent (bool): set to true to show the event file quicklook
        - bkgsubtract (bool): apply IRF background subtraction in skymap
        - smoothing_FWHM (quantity): apply smoothing to skymap
        - MapCenteredOnTarget (bool): to center the skymaps on target 
        or pointing

        """
        
        #----- Get the obs ID to run (defaults is all of them)        
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be quicklooked: '+str(obsID))

        #----- Show the observing properties
        if ShowObsDef:
            plotting.show_pointings(self.output_dir+'/Sim_ObsDef.xml',
                                    self.cluster, self.compact_source,
                                    self.output_dir+'/Sim_ObsPointing.pdf')
            plotting.show_obsdef(self.output_dir+'/Sim_ObsDef.xml',
                                 self.cluster.coord, self.output_dir+'/Sim_ObsDef.pdf')
            plotting.show_irf(self.obs_setup.caldb, self.obs_setup.irf, self.output_dir+'/Sim_ObsIRF')
        
        #----- Show the cluster model
        if ShowSkyModel:
            plotting.show_model_spectrum(self.output_dir+'/Sim_Model_Unstack.xml',
                                         self.output_dir+'/Sim_Model_Spectra.pdf')
            self._match_cluster_to_pointing()
            self.cluster.output_dir = self.output_dir+'/Sim_Model_Plots'
            if not os.path.exists(self.cluster.output_dir): os.mkdir(self.cluster.output_dir)
            self.cluster.plot()
            
        #----- Run the quicklook for eventfiles
        if ShowEvent:
            Nobs_done = 0
            for iobs in obsID:
                setup_obs = self.obs_setup.select_obs(iobs)

                if os.path.exists(self.output_dir+'/Events'+setup_obs.obsid[0]+'.fits'):
                    # Plot the events
                    plotting.events_quicklook(self.output_dir+'/Events'+setup_obs.obsid[0]+'.fits',
                                              self.output_dir+'/Events'+setup_obs.obsid[0]+'.pdf')
        
                    # Skymaps    
                    plotting.skymap_quicklook(self.output_dir+'/Sim_Skymap'+setup_obs.obsid[0],
                                              self.output_dir+'/Events'+setup_obs.obsid[0]+'.fits',
                                              setup_obs, self.compact_source, self.cluster,
                                              map_reso=self.cluster.map_reso,
                                              smoothing_FWHM=smoothing_FWHM,
                                              bkgsubtract=bkgsubtract,
                                              silent=self.silent, MapCenteredOnTarget=MapCenteredOnTarget)
                    
                else:
                    if not self.silent:
                        print(self.output_dir+'/Events'+setup_obs.obsid[0]+'.fits does not exist')
                        print('--> no quicklook available')
    
                # Information
                Nobs_done += 1
                if not self.silent: print('----- Quicklook '+str(Nobs_done)+'/'+
                                          str(len(self.obs_setup.obsid))+' done')
            
