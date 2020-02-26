"""
This file contains the CTAana class. It is dedicated to run
modules that allow for a user dedicated analysis.

"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import astropy.units as u
import copy
import ctools
import gammalib

from ClusterPipe.Tools import tools_spectral
from ClusterPipe.Tools import tools_imaging
from ClusterPipe.Tools import tools_timing
from ClusterPipe.Tools import plotting
from ClusterPipe.Tools import cubemaking


#==================================================
# Cluster class
#==================================================

class CTAana(object):
    """ 
    CTAana class. 
    This class serves as parser to the ClsuterPipe
    class to perform analysis.
    
    Methods
    ----------  
    - stack the data
    - compute residual map
    - compute profile
    - plots
    - ...

    """
    
    #==================================================
    # Run the baseline data analysis
    #==================================================
    
    def run_analysis(self,
                     obsID=None,
                     UsePtgRef=True,
                     refit=False,
                     like_accuracy=0.005,
                     max_iter=50,
                     fix_spat_for_ts=False):
        """
        Run the standard cluster analysis.
        
        Parameters
        ----------
        
        """

        #----- Data preparation
        self.run_ana_dataprep(obsID=obsID, UsePtgRef=UsePtgRef)
        
        #----- Likelihood fit
        self.run_ana_likelihood(refit=refit, like_accuracy=like_accuracy,
                                max_iter=max_iter, fix_spat_for_ts=fix_spat_for_ts)

        #----- Timing analysis
        self.run_ana_spectral()
        
        #----- Spectral analysis
        self.run_ana_spectral()
        
        #----- Imaging analysis
        self.run_ana_imaging()

        #----- Output plots
        self.run_ana_plots()
        
        
    #==================================================
    # Data preparation
    #==================================================
    
    def run_ana_dataprep(self, obsID=None, UsePtgRef=True):
        """
        This function is used to prepare the data to the 
        analysis.
        
        Parameters
        ----------
        - UsePtgRef (bool): use this keyword to match the
        coordinates of the cluster template, map coordinates
        and FoV
        
        """
        
        #----- Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
        
        #----- Get the obs ID to run
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be analysed: '+str(obsID))
        self.obs_setup.match_bkg_id() # make sure Bkg are unique
        
        #----- Deal with coordinates
        if UsePtgRef:
            self._match_cluster_to_pointing()
            self.map_coord = copy.deepcopy(self.cluster.map_coord)
            self.map_fov = np.amax(self.cluster.map_fov.to_value('deg'))*u.deg

        #----- Observation definition file
        self.obs_setup.write_pnt(self.output_dir+'/AnaPnt.def', obsid=obsID)
        self.obs_setup.run_csobsdef(self.output_dir+'/AnaPnt.def', self.output_dir+'/AnaObsDef.xml')

        #----- Get the events xml file
        xml     = gammalib.GXml(self.output_dir+'/Events.xml')
        obslist = xml.element('observation_list')
        obsid_in = []
        for i in range(len(obslist)):
            if obslist[i].attribute('id') not in obsID:
                obslist.remove(i)
            else:
                obsid_in.append(obslist[i].attribute('id'))
        for i in range(len(obsID)):
            if obsID[i] not in obsid_in:
                print('WARNING: Event file with obsID '+obsID[i]+' does not exist. It is ignored.')
        xml.save(self.output_dir+'/AnaEvents.xml')
        
        #----- Data selection
        sel = ctools.ctselect()
        sel['inobs']  = self.output_dir+'/AnaEvents.xml'
        sel['outobs'] = self.output_dir+'/AnaEventsSelected.xml'
        sel['prefix'] = self.output_dir+'/AnaSelected'
        sel['rad']    = self.map_fov.to_value('deg')
        sel['ra']     = self.map_coord.icrs.ra.to_value('deg')
        sel['dec']    = self.map_coord.icrs.dec.to_value('deg')
        sel['emin']   = self.spec_emin.to_value('TeV')
        sel['emax']   = self.spec_emax.to_value('TeV')
        if self.time_tmin is not None:
            sel['tmin'] = self.time_tmin
        else:
            sel['tmin'] = 'NONE'
        if self.time_tmax is not None:
            sel['tmax'] = self.time_tmax
        else:
            sel['tmax'] = 'NONE'
        sel.execute()
        print(sel)

        #----- Model
        self._make_model(prefix='AnaModelInput')

        #----- Binning
        if self.method_binned:
            cubemaking.counts_cube(self.output_dir,
                                   self.map_reso, self.map_coord, self.map_fov,
                                   self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                   stack=self.method_stack)
            if self.method_stack:
                cubemaking.exp_cube(self.output_dir,
                                    self.map_reso, self.map_coord, self.map_fov,
                                    self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg)
                cubemaking.psf_cube(self.output_dir,
                                    self.map_reso, self.map_coord, self.map_fov,
                                    self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg)
                cubemaking.bkg_cube(self.output_dir)
                if self.spec_edisp:
                    cubemaking.edisp_cube(self.output_dir,
                                          self.map_coord, self.map_fov,
                                          self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg)

        
    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_likelihood(self,
                           refit=False,
                           like_accuracy=0.005,
                           max_iter=50,
                           fix_spat_for_ts=False):
        """
        Run the likelihood analysis
        
        Parameters
        ----------
        
        """
        
        like = ctools.ctlike()
        
        # Input event list, counts cube or observation definition XML file.
        if self.method_binned:
            if self.method_stack:
                like['inobs']    = self.output_dir+'/AnaCountscube.fits'
            else:
                like['inobs']    = self.output_dir+'/AnaCountscube.xml'
        else:
            like['inobs']    = self.output_dir+'/AnaEventsSelected.xml'

        # Input model XML file.
        if self.method_binned and self.method_stack:
            like['inmodel']  = self.output_dir+'/AnaModelIntputStack.xml'
        else:
            like['inmodel']  = self.output_dir+'/AnaModelInput.xml'
        
        # Input exposure cube file.
        if self.method_binned and self.method_stack :
            like['expcube']  = self.output_dir+'/AnaExpcube.fits'
            
        # Input PSF cube file
        if self.method_binned and self.method_stack :
            like['psfcube']  = self.output_dir+'/AnaPsfcube.fits'
            
        # Input background cube file.
        if self.method_binned and self.method_stack :
            like['bkgcube']  = self.output_dir+'/AnaBkgcube.fits'
            
        # Input energy dispersion cube file.
        if self.method_binned and self.method_stack and self.spec_edisp:
            like['edispcube']  = self.output_dir+'/AnaEdispcube.fits'
        
        # Calibration database.
        #like['caldb']  =
        
        # Instrument response function.
        #like['irf']  = 
        
        # Applies energy dispersion to response computation.
        like['edisp']  = self.spec_edisp

        # Output model XML file with values and uncertainties updated by the maximum likelihood fit.
        like['outmodel'] = self.output_dir+'/AnaModelOutput.xml'

        # Output FITS or CSV file to store covariance matrix.
        like['outcovmat']  = self.output_dir+'/AnaModelOutputCovmat.fits'

        # Optimization statistic. 
        like['statistic']  = self.method_stat

        # Perform refitting of solution after initial fit.
        like['refit']  = refit

        # Absolute accuracy of maximum likelihood value.
        like['like_accuracy']  = like_accuracy

        # Maximum number of fit iterations.
        like['max_iter']  = max_iter

        # Fix spatial parameters for TS computation.
        like['fix_spat_for_ts']  = fix_spat_for_ts
        
        like.execute()
        print(like.obs())
        print(like.opt())

        
    #==================================================
    # Timing analysis
    #==================================================
    
    def run_ana_timing(self):
        """
        Run the timing analysis
        
        Parameters
        ----------
        
        """
        
        Nsource = xxx

        for isource in range(Nsource):
        
            #----- Compute lightcurve
            tools_timing.lightcurve()

            #----- Compute lightcurve
            tools_timing.find_variability()
        

    #==================================================
    # Run the spectral analysis
    #==================================================
    
    def run_ana_spectral(self):
        """
        Run the spectral analysis
        
        Parameters
        ----------
        
        """

        Nsource = xxx

        for isource in range(Nsource):
        
            #----- Compute spectra
            tools_spectral.spectrum()
            
            #----- Compute residual
            tools_spectral.residual()
            
            #----- Compute butterfly
            tools_spectral.butterfly()
        
        
    #==================================================
    # Run the imaging analysis
    #==================================================
    
    def run_ana_imaging(self):
        """
        Run the imaging analysis
        
        Parameters
        ----------
        
        """

        #----- Compute skymap
        tools_imaging.skymap()

        #----- Search for sources
        tools_imaging.src_detect()
        
        #----- Compute residual (w/wo cluster subtracted)
        tools_imaging.residual()
        
        #----- Compute profile
        tools_imaging.profile()

        
    #==================================================
    # Run the plotting tools
    #==================================================
    
    def run_ana_plot(self, obsID=None):
        """
        Run the plots
        
        Parameters
        ----------
        
        """

        #----- Get the obs ID to run (defaults is all of them)
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be looked at: '+str(obsID))

        #----- Plot the observing properties
        plotting.show_pointings(self.output_dir+'/AnaObsDef.xml', self.output_dir+'/AnaObsPointing.png')
        plotting.show_obsdef(self.output_dir+'/AnaObsDef.xml', self.cluster.coord, self.output_dir+'/AnaObsDef.png')
        plotting.show_irf(self.obs_setup.caldb, self.obs_setup.irf, self.output_dir+'/AnaObsIRF')
        
        #----- Show events
        for iobs in obsID:
            if os.path.exists(self.output_dir+'/AnaSelectedEvents'+self.obs_setup.select_obs(iobs).obsid[0]+'.fits'):
                plotting.events_quicklook(self.output_dir+'/AnaSelectedEvents'+self.obs_setup.select_obs(iobs).obsid[0]+'.fits',
                                          self.output_dir+'/AnaSelectedEvents'+self.obs_setup.select_obs(iobs).obsid[0]+'.png')
                
                from clustpipe_sim_plot import skymap_quicklook
                skymap_quicklook(self.output_dir+'/AnaSkymap'+self.obs_setup.select_obs(iobs).obsid[0],
                                 self.output_dir+'/AnaSelectedEvents'+self.obs_setup.select_obs(iobs).obsid[0]+'.fits',
                                 self.obs_setup.select_obs(iobs), self.compact_source, self.cluster,
                                 map_reso=self.cluster.map_reso, bkgsubtract=False,
                                 silent=True, MapCenteredOnTarget=True)
