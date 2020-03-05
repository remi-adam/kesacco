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
from ClusterPipe.Tools import utilities
from clustpipe_sim_plot import skymap_quicklook


#==================================================
# Cluster class
#==================================================

class CTAana(object):
    """ 
    CTAana class. 
    This class serves as parser to the ClusterPipe
    class to perform analysis.
    
    Methods
    ----------  
    - run_analysis
    - run_ana_dataprep
    - run_ana_likelihood
    - run_ana_imaging
    - run_ana_timing
    - run_ana_spectral
    - run_ana_plot

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

        #----- Imaging analysis
        self.run_ana_imaging()
        
        #----- Spectral analysis
        #self.run_ana_spectral()

        #----- Timing analysis
        #self.run_ana_timing()
        
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
        - obsID (str): list of obsID to be used in data preparation. 
        By default, all of the are used.
        - UsePtgRef (bool): use this keyword to match the coordinates 
        of the cluster template, map coordinates and FoV to pointings.
        
        """
        
        #----- Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
        
        #----- Get the obs ID to run
        obsID = self._check_obsID(obsID)
        if not self.silent: print('----- ObsID to be analysed: '+str(obsID))
        self.obs_setup.match_bkg_id() # make sure Bkg are unique
        
        #----- Observation definition file
        self.obs_setup.write_pnt(self.output_dir+'/Ana_Pnt.def', obsid=obsID)
        self.obs_setup.run_csobsdef(self.output_dir+'/Ana_Pnt.def', self.output_dir+'/Ana_ObsDef.xml')

        #----- Get the events xml file for the considered obsID
        self._write_new_xmlevent_from_obsid(self.output_dir+'/Events.xml', self.output_dir+'/Ana_Events.xml', obsID)
        
        #----- Data selection
        sel = ctools.ctselect()
        sel['inobs']  = self.output_dir+'/Ana_Events.xml'
        sel['outobs'] = self.output_dir+'/Ana_EventsSelected.xml'
        sel['prefix'] = self.output_dir+'/Ana_Selected'
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
            
        if not self.silent:
            print(sel)

        sel.execute()
        
        #----- Model
        if UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings
            
        self._make_model(prefix='Ana_Model_Input') # Compute the model files

        #----- Binning
        if self.method_binned:
            ctscube = cubemaking.counts_cube(self.output_dir,
                                             self.map_reso, self.map_coord, self.map_fov,
                                             self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                             stack=self.method_stack, silent=self.silent)
            if self.method_stack:
                expcube = cubemaking.exp_cube(self.output_dir,
                                              self.map_reso, self.map_coord, self.map_fov,
                                              self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                              silent=self.silent)
                psfcube = cubemaking.psf_cube(self.output_dir,
                                              self.map_reso, self.map_coord, self.map_fov,
                                              self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                              silent=self.silent)
                bkgcube = cubemaking.bkg_cube(self.output_dir, silent=self.silent)
                if self.spec_edisp:
                    edcube = cubemaking.edisp_cube(self.output_dir,
                                                   self.map_coord, self.map_fov,
                                                   self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                                   silent=self.silent)
                    
                    
    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_likelihood(self, UsePtgRef=True,
                           refit=False,
                           like_accuracy=0.005,
                           max_iter=50,
                           fix_spat_for_ts=False):
        """
        Run the likelihood analysis.
        See http://cta.irap.omp.eu/ctools/users/reference_manual/ctlike.html
        
        Parameters
        ----------
        - UsePtgRef (bool): use this keyword to match the coordinates 
        of the cluster template, map coordinates and FoV to pointings.
        - refit (bool): Perform refitting of solution after initial fit.
        - like_accuracy (float): Absolute accuracy of maximum likelihood value
        - max_iter (int): Maximum number of fit iterations.
        - fix_spat_for_ts (bool): Fix spatial parameters for TS computation.
        
        """

        #----- Make sure the map definition is ok
        if UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings

        #========== Run the likelihood
        if not self.silent:
            if (not self.method_binned) and self.method_stack:
                print('WARNING: unbinned likelihood are not stacked')
        
        like = ctools.ctlike()
        
        # Input event list, counts cube or observation definition XML file.
        if self.method_binned:
            if self.method_stack:
                like['inobs']    = self.output_dir+'/Ana_Countscube.fits'
            else:
                like['inobs']    = self.output_dir+'/Ana_Countscube.xml'
        else:
            like['inobs']    = self.output_dir+'/Ana_EventsSelected.xml'

        # Input model XML file.
        if self.method_binned and self.method_stack:
            like['inmodel']  = self.output_dir+'/Ana_Model_Intput_Stack.xml'
        else:
            like['inmodel']  = self.output_dir+'/Ana_Model_Input.xml'
        
        # Input exposure cube file.
        if self.method_binned and self.method_stack :
            like['expcube']  = self.output_dir+'/Ana_Expcube.fits'
            
        # Input PSF cube file
        if self.method_binned and self.method_stack :
            like['psfcube']  = self.output_dir+'/Ana_Psfcube.fits'
            
        # Input background cube file.
        if self.method_binned and self.method_stack :
            like['bkgcube']  = self.output_dir+'/Ana_Bkgcube.fits'
            
        # Input energy dispersion cube file.
        if self.method_binned and self.method_stack and self.spec_edisp:
            like['edispcube']  = self.output_dir+'/Ana_Edispcube.fits'
        
        # Calibration database
        #like['caldb']  =
        
        # Instrument response function
        #like['irf']  = 
        
        # Applies energy dispersion to response computation.
        like['edisp']  = self.spec_edisp

        # Output model XML file with values and uncertainties updated by the maximum likelihood fit.
        like['outmodel'] = self.output_dir+'/Ana_Model_Output.xml'

        # Output FITS or CSV file to store covariance matrix.
        like['outcovmat']  = self.output_dir+'/Ana_Model_Output_Covmat.fits'

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
        
        if not self.silent:
            print(like.opt())
            print(like.obs())
            print(like.obs().models())

        #========== Compute the binned model
        modcube = cubemaking.model_cube(self.output_dir,
                                        self.map_reso, self.map_coord, self.map_fov,
                                        self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                        edisp=self.spec_edisp,
                                        stack=self.method_stack, silent=self.silent)
            
            
    #==================================================
    # Run the imaging analysis
    #==================================================
    
    def run_ana_imaging(self, UsePtgRef=True, bkgsubtract='NONE', do_TS=False):
        """
        Run the imaging analysis
        
        Parameters
        ----------
        
        """

        #----- Make sure the map definition is ok
        if UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings
            
        npix = utilities.npix_from_fov_def(self.map_fov, self.map_reso)
        
        #----- Compute skymap
        skymap = tools_imaging.skymap(self.output_dir+'/Ana_EventsSelected.xml',
                                      self.output_dir+'/Ana_SkymapTot.fits',
                                      npix,self.map_reso.to_value('deg'),
                                      self.map_coord.icrs.ra.to_value('deg'),
                                      self.map_coord.icrs.dec.to_value('deg'),
                                      emin=self.spec_emin.to_value('TeV'), 
                                      emax=self.spec_emax.to_value('TeV'),
                                      caldb=None, irf=None,
                                      bkgsubtract=bkgsubtract,
                                      roiradius=0.1,inradius=1.0,outradius=2.0,
                                      iterations=3,threshold=3,
                                      silent=self.silent)
        
        #----- Search for sources
        srcmap = tools_imaging.src_detect(self.output_dir+'/Ana_SkymapTot.fits',
                                          self.output_dir+'/Ana_Sourcedetect.xml',
                                          self.output_dir+'/Ana_Sourcedetect.reg',
                                          threshold=4.0, maxsrcs=20, avgrad=1.0, exclrad=0.2,
                                          silent=self.silent)
        
        #----- Compute residual (w/wo cluster subtracted)
        resmap = tools_imaging.residual()

        #----- Compute the TS map
        if do_TS:
            expcube   = None
            psfcube   = None
            bkgcube   = None
            edispcube = None
            if self.method_binned:
                if self.method_stack:
                    inobs    = self.output_dir+'/Ana_Countscube.fits'
                    inmodel  = self.output_dir+'/Ana_Model_Intput_Stack.xml'
                    expcube  = self.output_dir+'/Ana_Expcube.fits'
                    psfcube  = self.output_dir+'/Ana_Psfcube.fits'
                    bkgcube  = self.output_dir+'/Ana_Bkgcube.fits'
                    if self.spec_edisp:
                        edispcube = self.output_dir+'/Ana_Edispcube.fits'
                else:
                    inobs   = self.output_dir+'/Ana_Countscube.xml'
                    inmodel = self.output_dir+'/Ana_Model_Input.xml'
            else:
                inobs   = self.output_dir+'/Ana_EventsSelected.xml'
                inmodel = self.output_dir+'/Ana_Model_Input.xml'
                
            tsmap = tools_imaging.tsmap(inobs, inmodel, self.output_dir+'/Ana_TSmap_IC310.fits',
                                        'IC310',
                                        npix, self.map_reso.to_value('deg'),
                                        self.map_coord.icrs.ra.to_value('deg'),
                                        self.map_coord.icrs.dec.to_value('deg'),
                                        expcube=expcube, psfcube=psfcube, bkgcube=bkgcube, edispcube=edispcube,
                                        caldb=None, irf=None, edisp=self.spec_edisp,
                                        statistic=self.method_stat)
        
        #----- Compute profile
        #tools_imaging.profile()


            
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
    # Run the plotting tools
    #==================================================
    
    def run_ana_plot(self):
        """
        Run the plots
        
        Parameters
        ----------
        
        """
        
        #----- Get the obs ID to run (defaults is all of them)
        obsID = self.obs_setup.obsid
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
                
                skymap_quicklook(self.output_dir+'/AnaSkymap'+self.obs_setup.select_obs(iobs).obsid[0],
                                 self.output_dir+'/AnaSelectedEvents'+self.obs_setup.select_obs(iobs).obsid[0]+'.fits',
                                 self.obs_setup.select_obs(iobs), self.compact_source, self.cluster,
                                 map_reso=self.cluster.map_reso, bkgsubtract='NONE',
                                 silent=True, MapCenteredOnTarget=True)

        #----- Show Combined map
        ps_name  = []
        ps_ra    = []
        ps_dec   = []
        for i in range(len(self.compact_source.name)):
            ps_name.append(self.compact_source.name[i])
            ps_ra.append(self.compact_source.spatial[i]['param']['RA']['value'].to_value('deg'))
            ps_dec.append(self.compact_source.spatial[i]['param']['DEC']['value'].to_value('deg'))

        ptg_name  = []
        ptg_ra    = []
        ptg_dec   = []
        for i in range(len(self.obs_setup.name)):
            ptg_name.append(self.obs_setup.name[i])
            ptg_ra.append(self.obs_setup.coord[i].icrs.ra.to_value('deg'))
            ptg_dec.append(self.obs_setup.coord[i].icrs.dec.to_value('deg'))

        plotting.show_map(self.output_dir+'/AnaSkymapTot.fits',
                          self.output_dir+'/AnaSkymapTot.pdf',
                          smoothing_FWHM=0.1*u.deg,
                          cluster_ra=self.cluster.coord.icrs.ra.to_value('deg'),
                          cluster_dec=self.cluster.coord.icrs.dec.to_value('deg'),
                          cluster_t500=self.cluster.theta500.to_value('deg'),
                          cluster_name=self.cluster.name,
                          ps_name=ps_name,
                          ps_ra=ps_ra,
                          ps_dec=ps_dec,
                          ptg_ra=ptg_ra,
                          ptg_dec=ptg_dec,
                          #PSF=PSF_tot,
                          bartitle='Counts',
                          rangevalue=[None, None],
                          logscale=True,
                          significance=False,
                          cmap='magma')
