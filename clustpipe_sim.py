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
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates.sky_coordinate import SkyCoord
from random import randint
import gammalib
import ctools

from ClusterPipe.Tools import make_cluster_template
from ClusterPipe.Tools import plotting
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

        #----- Get the obs ID to run
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be observed: '+str(obsID))

        #----- Make sure the cluster FoV matches all requested observations
        self.match_cluster_to_pointing()
        
        #----- Make cluster templates
        self._make_model(prefix='SimModel')

        #----- Make observation files
        self.obs_setup.write_pnt(self.output_dir+'/SimPnt.def', obsid=obsID)
        self.obs_setup.run_csobsdef(self.output_dir+'/SimPnt.def', self.output_dir+'/SimObsDef.xml')

        #----- Get the seed for reapeatable simu
        if seed is None: seed = randint(1, 1e6)
        
        #----- Run the observation
        obssim = ctools.ctobssim()
        obssim['inobs']      = self.output_dir+'/SimObsDef.xml'
        obssim['inmodel']    = self.output_dir+'/SimModel.xml'
        obssim['prefix']     = self.output_dir+'/TmpEvents'
        obssim['outevents']  = self.output_dir+'/Events.xml'
        obssim['edisp']      = self.edisp
        obssim['startindex'] = 1
        obssim['maxrate']    = 1e6
        obssim['seed']       = seed
        obssim.execute()
        if not self.silent:
            print('------- Simulation log -------')
            print(obssim)
            print('')

        self._correct_eventfile_names(self.output_dir+'/Events.xml', prefix='Events')
        
        
    #==================================================
    # Quicklook
    #==================================================

    def run_sim_quicklook(self, obsID=None,
                          ShowSkyModel=True,
                          ShowEvent=True,
                          ShowObsDef=True,
                          bkgsubtract=None,
                          smoothing_FWHM=0.03*u.deg,
                          MapCenteredOnTarget=True):
        """
        Provide quicklook analysis of the simulation
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be observed
        - ShowObsDef (bool): set to true to show the observation definition
        - ShowCluster (bool): set to true to show the cluster model
        - ShowEvent (bool): set to true to show the event file quicklook
        - bkgsubtract (bool): apply IRF background subtraction in skymap
        - smoothing_FWHM (quantity): apply smoothing to skymap
        - MapCenteredOnTarget (bool): to center the skymaps on target 
        or pointing

        """
        
        #----- Get the obs ID to run (defaults is all of them)        
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be observed: '+str(obsID))

        #----- Show the observing properties
        if ShowObsDef:
            plotting.show_pointings(self.output_dir+'/SimObsDef.xml', self.output_dir+'/SimObsPointing.png')
            plotting.show_obsdef(self.output_dir+'/SimObsDef.xml', self.cluster.coord, self.output_dir+'/SimObsDef.png')
            plotting.show_irf(self.obs_setup.caldb, self.obs_setup.irf, self.output_dir+'/SimObsIRF')
        
        #----- Show the cluster model
        if ShowSkyModel:
            plotting.show_model_spectrum(self.output_dir+'/SimModel.xml', self.output_dir+'/SimModelSpectra.png')
            self.match_cluster_to_pointing()
            self.cluster.output_dir = self.output_dir+'/SimModelPlots'
            if not os.path.exists(self.cluster.output_dir): os.mkdir(self.cluster.output_dir)
            self.cluster.plot()
            
        #----- Run the quicklook for eventfiles
        if ShowEvent:
            Nobs_done = 0
            for iobs in obsID:
                clustpipe_sim_plot.main(self.output_dir,
                                        self.cluster,
                                        self.compact_source,
                                        self.obs_setup.select_obs(iobs),
                                        map_reso=self.cluster.map_reso,
                                        bkgsubtract=bkgsubtract,
                                        smoothing_FWHM=smoothing_FWHM,
                                        silent=self.silent,
                                        MapCenteredOnTarget=MapCenteredOnTarget)
                
                # Information
                Nobs_done += 1
                if not self.silent: print('----- Quicklook '+str(Nobs_done)+'/'+str(len(self.obs_setup.obsid))+' done')
            
    #==================================================
    # Correct XML and file names for simulated events
    #==================================================
    
    def _correct_eventfile_names(self, xmlfile, prefix='Events'):
        """
        Change the event filename and associated xml file
        by naming them using the obsid
        
        Parameters
        ----------
        - xmlfile (str): the xml file name
        
        """
        
        xml     = gammalib.GXml(xmlfile)
        obslist = xml.element('observation_list')
        for i in range(len(obslist)):
            obs = obslist[i]
            obsid = obs.attribute('id')

            # make sure the EventList key exist
            killnum = None
            for j in range(len(obs)):
                if obs[j].attribute('name') == 'EventList': killnum = j

            # In case there is one EventList, move the file and rename the xml
            if killnum is not None:
                file_in = obs[killnum].attribute('file')
                file_out = os.path.dirname(file_in)+'/'+prefix+obsid+'.fits'
                os.rename(file_in, file_out)
                obs.remove(killnum)
                obs.append('parameter name="EventList" file="'+file_out+'"')
                
        xml.save(xmlfile)      
        
