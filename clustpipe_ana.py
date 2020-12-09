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
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.time import Time
import copy
import ctools
import cscripts
import gammalib

from kesacco.Tools import tools_spectral
from kesacco.Tools import tools_imaging
from kesacco.Tools import tools_timing
from kesacco.Tools import plotting
from kesacco.Tools import cubemaking
from kesacco.Tools import utilities
from kesacco.Tools import build_ctools_model
from kesacco.Tools import mcmc_spectrum
from kesacco.Tools import mcmc_profile
from kesacco       import clustpipe_ana_plot

from minot.ClusterTools.map_tools import radial_profile_cts


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
    - run_analysis : run the full analysis (i.e. all the submodules one after the other)
    - run_ana_dataprep : run the data preparation, including writting observation files, 
    selecting the data, bining and stacking
    - run_ana_likelihood: likelihood fit of the sky model to the data
    - run_ana_sensitivity: compute the sensitivity curve for a given source model
    - run_ana_upperlimit: compute the upper limit for a given parameter
    - run_ana_imaging: perform the analysis related to the image (skymaps, residuals, source 
    detection ,TS map)
    - run_ana_spectral: perform the spectral analysis (spectrum and residuals of sources 
    in the ROI)
    - run_ana_timing: perform the time analysis (lightcurve of sources in the ROI)
    - run_ana_expected_output: compute the IRF convolved expected cluster based 
    on the input simulation
    - run_ana_mcmc: run the MCMC to constrain the spectrum and profile
    - run_ana_plot: run the plotting tools to show the results

    To do list
    ----------
    - Include on/off analysis
    - Move the MCMC plots into the plot section
    - Implement:
       - run_ana_sensitivity
       - run_ana_upperlimit

    """
    
    #==================================================
    # Run the baseline data analysis
    #==================================================
    
    def run_analysis(self,
                     obsID=None,
                     refit=False,
                     like_accuracy=0.005,
                     max_iter=50,
                     fix_spat_for_ts=False,
                     imaging_bkgsubtract='NONE',
                     do_Skymap=False,
                     do_SourceDet=False,
                     do_ResMap=False,
                     do_TSmap=False,
                     profile_reso=0.05*u.deg,
                     do_Spec=True,
                     do_Butterfly=True,
                     do_Resspec=True,
                     smoothing_FWHM=0.1*u.deg,
                     profile_log=True,
                     do_timing=False,
                     do_MCMC_spectrum=False,
                     do_MCMC_profile=False):
        """
        Run the standard cluster analysis. This pipeline runs
        all the sub-modules one by one according to the analysis definition
        parameters.
        
        Parameters
        ----------
        - Only parsing parameters, see individual sub-modules
        
        """

        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False
        
        #----- Data preparation
        self.run_ana_dataprep(obsID=obsID)
        
        #----- Likelihood fit
        self.run_ana_likelihood(refit=refit,
                                like_accuracy=like_accuracy,
                                max_iter=max_iter,
                                fix_spat_for_ts=fix_spat_for_ts)
        
        #----- Imaging analysis
        self.run_ana_imaging(bkgsubtract=imaging_bkgsubtract,
                             do_Skymap=do_Skymap,
                             do_SourceDet=do_SourceDet,
                             do_Res=do_ResMap,
                             do_TS=do_TSmap,
                             profile_reso=profile_reso)
        
        #----- Spectral analysis
        self.run_ana_spectral(do_Spec=do_Spec,
                              do_Butterfly=do_Butterfly,
                              do_Res=do_Resspec)

        #----- Timing analysis
        if do_timing:
            self.run_ana_timing()

        #----- Expected output computation
        self.run_ana_expected_output(profile_reso=profile_reso)

        #----- A posteriori MCMC fit for the spectrum and profile
        self.run_ana_mcmc(do_spectrum=do_MCMC_spectrum,
                          do_profile=do_MCMC_profile,
                          reset_mcmc=True, run_mcmc=True)
        
        #----- Output plots
        self.run_ana_plot(obsID=obsID,
                          smoothing_FWHM=smoothing_FWHM,
                          profile_log=profile_log)
        
        
    #==================================================
    # Data preparation
    #==================================================
    
    def run_ana_dataprep(self,
                         obsID=None):
        """
        This function is used to prepare the data to the 
        analysis.
        
        Parameters
        ----------
        - obsID (str): list of obsID to be used in data preparation. 
        By default, all of the are used.
        
        Outputs files
        -------------
        - Ana_Events.xml: observation list to be analysed
        - Ana_Pnt.def: observation definition file
        - Ana_ObsDef.xml: observation definition file as xml file
        - Ana_EventsSelected.xml: observation list to be analysed after selection
        - Ana_EventsSelected_log.txt: log file of the event selection
        - Ana_SelectedEvents{obsid}.fits: events file after data selection
        - Ana_{Psf,Exp,Counts,Bkg}cube.fits: data cube fits files
        - Ana_{Psf,Exp,Counts,Bkg}cube_log.txt: data cube log files
        - Ana_Countscube.xml: xml file gathering the counts cube for each obsid
        - Ana_Countscubecta_{obsid}.fits: counts cube for each obsid
        - Ana_Model_Input_Spectrum.txt: input model spectrum for the cluster
        - Ana_Model_Input_Map.fits: input map model for the cluster
        - Ana_Model_Input_Stack.xml: input sky model xml file for stacked analysis
        - Ana_Model_Input_Unstack.xml: input sky model xml file for unstacked analysis

        """

        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False
        
        #----- Create the output directory if needed
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        #----- Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings
            
        #----- Get the obs ID to run
        obsID = self._check_obsID(obsID)
        if not self.silent:
            print('----- ObsID to be analysed: '+str(obsID))
            print('')
        self.obs_setup.match_bkg_id() # make sure Bkg are unique
        
        #----- Observation definition file
        self.obs_setup.write_pnt(self.output_dir+'/Ana_Pnt.def',
                                 obsid=obsID)
        self.obs_setup.run_csobsdef(self.output_dir+'/Ana_Pnt.def',
                                    self.output_dir+'/Ana_ObsDef.xml')

        #----- Get the events xml file for the considered obsID
        self._write_new_xmlevent_from_obsid(self.output_dir+'/Events.xml',
                                            self.output_dir+'/Ana_Events.xml',
                                            obsID)
        
        #----- Data selection
        sel = ctools.ctselect()
        sel['inobs']    = self.output_dir+'/Ana_Events.xml'
        sel['outobs']   = self.output_dir+'/Ana_EventsSelected.xml'
        sel['prefix']   = self.output_dir+'/Ana_Selected'
        sel['usepnt']   = False
        sel['ra']       = self.map_coord.icrs.ra.to_value('deg')
        sel['dec']      = self.map_coord.icrs.dec.to_value('deg')
        sel['rad']      = self.map_fov.to_value('deg') * np.sqrt(2)/2.0
        sel['forcesel'] = True
        sel['emin']     = self.spec_emin.to_value('TeV')
        sel['emax']     = self.spec_emax.to_value('TeV')
        if self.time_tmin is not None:
            sel['tmin'] = self.time_tmin
        else:
            sel['tmin'] = 'NONE'
        if self.time_tmax is not None:
            sel['tmax'] = self.time_tmax
        else:
            sel['tmax'] = 'NONE'
        sel['phase']    = 'NONE'
        sel['expr']     = ''
        sel['usethres'] = 'NONE'
        sel['logfile']  = self.output_dir+'/Ana_EventsSelected_log.txt'
        sel['chatter']  = 2
        
        sel.logFileOpen()
        sel.execute()
        sel.logFileClose()

        if not self.silent:
            print(sel)
            print('')
        
        #----- Model
        map_template_fov = np.amin(self.cluster.map_fov.to_value('deg'))
        if (not self.silent) and (2*self.cluster.theta_truncation.to_value('deg') > map_template_fov):
            print('WARNING: the cluster extent is larger than the model map field of view.')
            print('         The recovered normalization will thus be biased low.')
            print('')
            
        self._make_model(prefix='Ana_Model_Input',
                         obsID=obsID) # Compute the model files

        #----- Binning (needed even if unbinned likelihood)
        for stacklist in [True, False]:
            ctscube = cubemaking.counts_cube(self.output_dir,
                                             self.map_reso, self.map_coord, self.map_fov,
                                             self.spec_emin, self.spec_emax,
                                             self.spec_enumbins, self.spec_ebinalg,
                                             stack=stacklist,
                                             logfile=self.output_dir+'/Ana_Countscube_log.txt',
                                             silent=self.silent)
        
        expcube = cubemaking.exp_cube(self.output_dir,
                                      self.map_reso, self.map_coord, self.map_fov,
                                      self.spec_emin, self.spec_emax,
                                      self.spec_enumbins, self.spec_ebinalg,
                                      logfile=self.output_dir+'/Ana_Expcube_log.txt',
                                      silent=self.silent)
        psfcube = cubemaking.psf_cube(self.output_dir,
                                      self.map_reso, self.map_coord, self.map_fov,
                                      self.spec_emin, self.spec_emax,
                                      self.spec_enumbins, self.spec_ebinalg,
                                      logfile=self.output_dir+'/Ana_Psfcube_log.txt',
                                      silent=self.silent)
        bkgcube = cubemaking.bkg_cube(self.output_dir,
                                      logfile=self.output_dir+'/Ana_Bkgcube_log.txt',
                                      silent=self.silent)
        
        if self.spec_edisp:
            edcube = cubemaking.edisp_cube(self.output_dir,
                                           self.map_coord, self.map_fov,
                                           self.spec_emin, self.spec_emax,
                                           self.spec_enumbins, self.spec_ebinalg,
                                           logfile=self.output_dir+'/Ana_Edispcube_log.txt',
                                           silent=self.silent)
        
        
    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_likelihood(self, refit=False,
                           like_accuracy=0.005,
                           max_iter=50,
                           fix_spat_for_ts=False):
        """
        Run the likelihood analysis.
        See http://cta.irap.omp.eu/ctools/users/reference_manual/ctlike.html
        
        Parameters
        ----------
        - refit (bool): Perform refitting of solution after initial fit.
        - like_accuracy (float): Absolute accuracy of maximum likelihood value
        - max_iter (int): Maximum number of fit iterations.
        - fix_spat_for_ts (bool): Fix spatial parameters for TS computation.
        
        Outputs files
        -------------
        - Ana_Model_Output.xml: constrained sky model
        - Ana_Model_Output_log.txt: log file of the likelihood fit
        - Ana_Model_Output_Covmat.fits: covariance matrix for the output fit
        - Ana_Model_Output_Cluster.xml: constrained sky model excluding the cluster
        - Ana_Model_Cube.fits: Best fit model cube for the all data
        - Ana_Model_Cube_log.txt: log file of the best fit model cube for the all data
        - Ana_Model_Cube_Cluster.fits: Best fit model cube for the all data without the cluster
        - Ana_Model_Cube_cluster_log.txt: log file of the best fit model cube for the all data without the cluster
        
        """

        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False
        
        #----- Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings

        #========== Run the likelihood
        if not self.silent:
            if (not self.method_binned) and self.method_stack:
                print('WARNING: unbinned likelihood are not stacked')
                print('')
        
        like = ctools.ctlike()
        
        # Input event list, counts cube or observation definition XML file.
        if self.method_binned:
            if self.method_stack:
                like['inobs'] = self.output_dir+'/Ana_Countscube.fits'
            else:
                like['inobs'] = self.output_dir+'/Ana_Countscube.xml'
        else:
            like['inobs']     = self.output_dir+'/Ana_EventsSelected.xml'

        # Input model XML file.
        if self.method_binned and self.method_stack:
            like['inmodel']  = self.output_dir+'/Ana_Model_Input_Stack.xml'
        else:
            like['inmodel']  = self.output_dir+'/Ana_Model_Input_Unstack.xml'
        
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

        # Log file
        like['logfile'] = self.output_dir+'/Ana_Model_Output_log.txt'

        like.logFileOpen()
        like.execute()
        like.logFileClose()
    
        if not self.silent:
            print(like.opt())
            print(like.obs())
            print(like.obs().models())
            print('')

        #========== Compute a fit model file without the cluster
        self._rm_source_xml(self.output_dir+'/Ana_Model_Output.xml',
                            self.output_dir+'/Ana_Model_Output_Cluster.xml',
                            self.cluster.name)

        #========== Compute the binned model
        modcube = cubemaking.model_cube(self.output_dir,
                                        self.map_reso, self.map_coord, self.map_fov,
                                        self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                        edisp=self.spec_edisp,
                                        stack=self.method_stack,
                                        logfile=self.output_dir+'/Ana_Model_Cube_log.txt',
                                        silent=self.silent)
            
        modcube_Cl = cubemaking.model_cube(self.output_dir,
                                           self.map_reso, self.map_coord, self.map_fov,
                                           self.spec_emin, self.spec_emax, self.spec_enumbins, self.spec_ebinalg,
                                           edisp=self.spec_edisp,
                                           stack=self.method_stack, silent=self.silent,
                                           logfile=self.output_dir+'/Ana_Model_Cube_Cluster_log.txt',
                                           inmodel_usr=self.output_dir+'/Ana_Model_Output_Cluster.xml',
                                           outmap_usr=self.output_dir+'/Ana_Model_Cube_Cluster.fits')

        
    #==================================================
    # Sensitivity
    #==================================================
    
    def run_ana_sensitivity(self,
                            tobs,
                            caldb='prod3b-v2', irf='North_z20_S_5h', deadc=0.95,
                            offset=0.0*u.deg, roi_rad=5*u.deg,
                            Nsigma=5,
                            NengPt=20,
                            max_iter=50,
                            source_name=None):
        """
        Get the sensitivity curve
        
        Parameters
        ----------
        - tobs (quantity): time of observation
        - caldb (str): calibration database
        - irf (str): instrument response function
        - deadc (float): deadtime fraction
        - offset (quantity): source offset from the center
        - roi_rad (quantity): radius of the ROI
        - Nsigma (float): number of sigma for the detection
        - NengPt (int): number of energy points to compute
        - max_iter (int): Maximum number of fit iterations
        - source_name (str): the name of the source

        Outputs files
        -------------
        - Ana_Sensitivity'+srcname+'.dat': the sensitivity curve
        
        """

        #----- Select the source name
        if source_name is None:
            srcname = self.cluster.name
        else:
            srcname = source_name

        #----- number of pixels for bining
        npix = utilities.npix_from_fov_def(self.map_fov, self.map_reso)

        #----- Fill the parameters and run
        sens = cscripts.cssens()

        sens['inobs']     = 'NONE' # Input event list, counts cube or observation definition XML file.
        sens['inmodel']   = self.output_dir+'/Ana_Model_Input_Unstack.xml'
        sens['srcname']   = srcname
        sens['caldb']     = caldb
        sens['irf']       = irf
        sens['edisp']     = self.spec_edisp
        sens['deadc']     = deadc
        sens['outfile']   = self.output_dir+'/Ana_Sensitivity'+srcname+'.dat'
        sens['offset']    = offset.to_value('deg') # Offset angle of source in field of view (in degrees).
        sens['duration']  = tobs.to_value('s')
        sens['rad']       = roi_rad.to_value('deg') # radius of ROI
        sens['emin']      = self.spec_emin.to_value('TeV')
        sens['emax']      = self.spec_emax.to_value('TeV')
        sens['bins']      = NengPt # number of bins for sensitivity computation
        sens['enumbins']  = self.spec_enumbins # number of bins for binned analysis
        sens['npix']      = npix
        sens['binsz']     = self.map_reso.to_value('deg')
        sens['type']      = 'Differential'
        sens['sigma']     = Nsigma
        sens['max_iter']  = max_iter
        sens['statistic'] = self.method_stat
        sens['logfile']   = self.output_dir+'/Ana_Sensitivity'+srcname+'_log.txt'

        sens.logFileOpen()
        sens.execute()
        sens.logFileClose()
        
        if not self.silent:
            print(sens)
            print('')
        
            
    #==================================================
    # Upper limit
    #==================================================
    
    def run_ana_upperlimit(self):
        """
        Get the upper limit on the fit parameters
        
        Parameters
        ----------
        
        """

        

        
    #==================================================
    # Run the imaging analysis
    #==================================================
    
    def run_ana_imaging(self, bkgsubtract='NONE',
                        do_Skymap=False,
                        do_SourceDet=False,
                        do_Res=False,
                        do_TS=False,
                        profile_reso=0.05*u.deg):
        """
        Run the imaging analysis
        
        Parameters
        ----------
        - bkgsubtract (str): apply background subtraction to skymap
        - do_Skymap (bool): compute skymap
        - do_SourceDet (bool): compute source detection
        - do_Res (bool): compute residual
        - do_TS (bool): compute TS map
        - profile_reso (quantity): bin size for profile
                
        Outputs files
        -------------
        - Ana_SkymapTot.fits: total sky map 
        - Ana_SkymapTot_log.txt: total sky map log
        - Ana_Sourcedetect.xml: source detection xml file
        - Ana_Sourcedetect.reg: source detection DS9 regions
        - Ana_Sourcedetect_log.txt: source detection log file
        - Ana_ResmapTot_{*}.fits: residual skymap
        - Ana_TSmap_{*}_log.txt: TS map log file
        - Ana_TSmap_{*}_log.fits: TS map fits file
        - Ana_ResmapCluster_profile.fits: profile of the residual skymap
        
        """

        #========== Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings
            
        npix = utilities.npix_from_fov_def(self.map_fov, self.map_reso)
        
        #========== Defines cubes
        inobs, inmodel, expcube, psfcube, bkgcube, edispcube, modcube, modcubeCl = self._define_std_filenames()
        
        #========== Compute skymap
        if do_Skymap:
            skymap = tools_imaging.skymap(self.output_dir+'/Ana_EventsSelected.xml',
                                          self.output_dir+'/Ana_SkymapTot.fits',
                                          npix, self.map_reso.to_value('deg'),
                                          self.map_coord.icrs.ra.to_value('deg'),
                                          self.map_coord.icrs.dec.to_value('deg'),
                                          emin=self.spec_emin.to_value('TeV'), 
                                          emax=self.spec_emax.to_value('TeV'),
                                          caldb=None, irf=None,
                                          bkgsubtract=bkgsubtract,
                                          roiradius=0.1, inradius=0.6, outradius=0.8,
                                          iterations=3, threshold=3,
                                          logfile=self.output_dir+'/Ana_SkymapTot_log.txt',
                                          silent=self.silent)
        
        #========== Search for sources
        if do_SourceDet:
            if os.path.isfile(self.output_dir+'/Ana_SkymapTot.fits'):
                srcmap = tools_imaging.src_detect(self.output_dir+'/Ana_SkymapTot.fits',
                                                  self.output_dir+'/Ana_Sourcedetect.xml',
                                                  self.output_dir+'/Ana_Sourcedetect.reg',
                                                  threshold=4.0, maxsrcs=10, avgrad=1.0,
                                                  corr_rad=0.05, exclrad=0.2,
                                                  logfile=self.output_dir+'/Ana_Sourcedetect_log.txt',
                                                  silent=self.silent)
            else:
                print(self.output_dir+'/Ana_SkymapTot.fits what not created and is needed for source detection.')
                print('')
                
        #========== Compute residual (w/wo cluster subtracted)
        if do_Res:
            #----- Total residual and keeping the cluster
            for alg in ['SIGNIFICANCE', 'SUB', 'SUBDIV']:
                resmap = tools_imaging.resmap(self.output_dir+'/Ana_Countscube.fits',
                                              self.output_dir+'/Ana_Model_Output.xml',
                                              self.output_dir+'/Ana_ResmapTot_'+alg+'.fits',
                                              npix, self.map_reso.to_value('deg'),
                                              self.map_coord.icrs.ra.to_value('deg'),
                                              self.map_coord.icrs.dec.to_value('deg'),
                                              emin=self.spec_emin.to_value('TeV'),
                                              emax=self.spec_emax.to_value('TeV'),
                                              enumbins=self.spec_enumbins, ebinalg=self.spec_ebinalg,
                                              modcube=modcube,
                                              expcube=expcube, psfcube=psfcube,
                                              bkgcube=bkgcube, edispcube=edispcube,
                                              caldb=None, irf=None,
                                              edisp=self.spec_edisp,
                                              algo=alg,
                                              logfile=self.output_dir+'/Ana_ResmapTot_'+alg+'_log.txt',
                                              silent=self.silent)

                resmap = tools_imaging.resmap(self.output_dir+'/Ana_Countscube.fits',
                                              self.output_dir+'/Ana_Model_Output_Cluster.xml',
                                              self.output_dir+'/Ana_ResmapCluster_'+alg+'.fits',
                                              npix, self.map_reso.to_value('deg'),
                                              self.map_coord.icrs.ra.to_value('deg'),
                                              self.map_coord.icrs.dec.to_value('deg'),
                                              emin=self.spec_emin.to_value('TeV'),
                                              emax=self.spec_emax.to_value('TeV'),
                                              enumbins=self.spec_enumbins, ebinalg=self.spec_ebinalg,
                                              modcube=modcubeCl,
                                              expcube=expcube, psfcube=psfcube,
                                              bkgcube=bkgcube, edispcube=edispcube,
                                              caldb=None, irf=None,
                                              edisp=self.spec_edisp,
                                              algo=alg,
                                              logfile=self.output_dir+'/Ana_ResmapCluster_'+alg+'_log.txt',
                                              silent=self.silent)

            #----- Cluster profile
            hdul       = fits.open(self.output_dir+'/Ana_ResmapCluster_SUB.fits')
            res_counts = hdul[0].data
            header     = hdul[0].header
            hdul.close()
            hdul       = fits.open(self.output_dir+'/Ana_ResmapTot_SUB.fits')
            res_all    = hdul[0].data
            header     = hdul[0].header
            hdul.close()
            hdul       = fits.open(self.output_dir+'/Ana_ResmapTot_SUBDIV.fits')
            subdiv_all = hdul[0].data
            header     = hdul[0].header
            hdul.close()
            model = res_all/subdiv_all
            
            radius, prof, err = radial_profile_cts(res_counts,
                                                   [self.cluster.coord.icrs.ra.to_value('deg'),
                                                    self.cluster.coord.icrs.dec.to_value('deg')],
                                                   stddev=np.sqrt(model), header=header,
                                                   binsize=profile_reso.to_value('deg'), stat='POISSON',
                                                   counts2brightness=True)
            tab  = Table()
            tab['radius']  = Column(radius, unit='deg',
                                    description='Cluster offset (bin='+str(profile_reso.to_value('deg'))+'deg')
            tab['profile'] = Column(prof,   unit='deg-2', description='Counts per deg-2')
            tab['error']   = Column(err,    unit='deg-2', description='Counts per deg-2 uncertainty')
            tab.write(self.output_dir+'/Ana_ResmapCluster_profile.fits', overwrite=True)
            
        #----- Compute the TS map
        if do_TS:
            fov_ts = 0.5*u.deg
            reso_ts = 0.05*u.deg
            npix_ts = utilities.npix_from_fov_def(fov_ts, reso_ts)

            for src in self.compact_source.name:
                wsrc = np.where(np.array(self.compact_source.name) == src)[0][0]
                ctr_ra  = self.compact_source.spatial[wsrc]['param']['RA']['value'].to_value('deg')
                ctr_dec = self.compact_source.spatial[wsrc]['param']['DEC']['value'].to_value('deg')
                tsmap = tools_imaging.tsmap(inobs, inmodel, self.output_dir+'/Ana_TSmap_'+src+'.fits',
                                            src, npix_ts, reso_ts.to_value('deg'), ctr_ra, ctr_dec,
                                            expcube=None, psfcube=None, bkgcube=None, edispcube=None,
                                            caldb=None, irf=None, edisp=self.spec_edisp,
                                            statistic=self.method_stat,
                                            logfile=self.output_dir+'/Ana_TSmap_'+src+'_log.txt',
                                            silent=self.silent)
                
                
    #==================================================
    # Run the spectral analysis
    #==================================================
    
    def run_ana_spectral(self, do_Spec=True, do_Butterfly=True, do_Res=True):
        """
        Run the spectral analysis
        
        Parameters
        ----------
        - do_Spec (bool): compute spectra
        - do_Butterfly (bool): compute butterfly
        - do_Res (bool): compute residual spectra
        
        Outputs files
        -------------
        - Ana_Spectrum_{*}.fits: spectrum file for the sources listed in the ROI
        - Ana_Spectrum_{*}_log.txt: spectrum logfile for the sources listed in the ROI
        - Ana_Spectrum_Buterfly_{*}.txt: butterfly file for the sources listed in the ROI
        - Ana_Spectrum_Buterfly_{*}_log.txt: butterfly log file for the sources listed in the ROI
        - Ana_Spectrum_Residual.fits: residual spectrum
        - Ana_Spectrum_Residual_log.txt: residual spectrum logfile
        
        """

        #========== Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings
            
        npix = utilities.npix_from_fov_def(self.map_fov, self.map_reso)
        
        #========== Defines cubes
        inobs, inmodel, expcube, psfcube, bkgcube, edispcube, modcube, modcubeCl = self._define_std_filenames()

        #========== Defines the sources
        models = gammalib.GModels(self.output_dir+'/Ana_Model_Output.xml')
        Nsource = len(models)
        
        for isource in range(Nsource):
            
            srcname = models[isource].name()

            if models[isource].type() not in ['CTACubeBackground', 'CTAIrfBackground']:
                if not self.silent:
                    print('--- Computing spectrum: '+srcname)

                #----- Compute spectra
                if do_Spec:
                    try:
                        tools_spectral.spectrum(inobs,  self.output_dir+'/Ana_Model_Output.xml',
                                                srcname, self.output_dir+'/Ana_Spectrum_'+srcname+'.fits',
                                                emin=self.spec_emin.to_value('TeV'),
                                                emax=self.spec_emax.to_value('TeV'),
                                                enumbins=self.spec_enumbins, ebinalg=self.spec_ebinalg,
                                                expcube=expcube,
                                                psfcube=psfcube,
                                                bkgcube=bkgcube,
                                                edispcube=edispcube,
                                                caldb=None,
                                                irf=None,
                                                edisp=self.spec_edisp,
                                                method='SLICE',
                                                statistic=self.method_stat,
                                                calc_ts=True,
                                                calc_ulim=True,
                                                fix_srcs=True,
                                                fix_bkg=False,
                                                logfile=self.output_dir+'/Ana_Spectrum_'+srcname+'_log.txt',
                                                silent=self.silent)
                    except ZeroDivisionError:
                        print(srcname+' spectrum not computed due to ZeroDivisionError')

                #----- Compute butterfly
                if do_Butterfly:
                    tools_spectral.butterfly(inobs, self.output_dir+'/Ana_Model_Output.xml',
                                             srcname, self.output_dir+'/Ana_Spectrum_Buterfly_'+srcname+'.txt',
                                             emin=5e-3, emax=5e+3,
                                             enumbins=100, ebinalg=self.spec_ebinalg,
                                             expcube=expcube,
                                             psfcube=psfcube,
                                             bkgcube=bkgcube,
                                             edispcube=edispcube,
                                             caldb=None,
                                             irf=None,
                                             edisp=self.spec_edisp,
                                             fit=False,
                                             method='GAUSSIAN',
                                             confidence=0.68,
                                             statistic=self.method_stat,
                                             like_accuracy=0.005,
                                             max_iter=50,
                                             matrix='NONE', #self.output_dir+'/Ana_Model_Output_Covmat.fits',
                                             logfile=self.output_dir+'/Ana_Spectrum_Buterfly_'+srcname+'_log.txt',
                                             silent=self.silent)

        #----- Compute residual in R500
        if do_Res:
            tools_spectral.residual(inobs, self.output_dir+'/Ana_Model_Output.xml',
                                    self.output_dir+'/Ana_Spectrum_Residual.fits',
                                    npix, self.map_reso.to_value('deg'),
                                    self.map_coord.icrs.ra.to_value('deg'),
                                    self.map_coord.icrs.dec.to_value('deg'),
                                    res_ra=self.cluster.coord.icrs.ra.to_value('deg'),
                                    res_dec=self.cluster.coord.icrs.dec.to_value('deg'),
                                    res_rad=self.cluster.theta500.to_value('deg'),
                                    emin=self.spec_emin.to_value('TeV'),
                                    emax=self.spec_emax.to_value('TeV'),
                                    enumbins=self.spec_enumbins, ebinalg=self.spec_ebinalg,
                                    modcube=modcube,
                                    expcube=expcube, psfcube=psfcube,
                                    bkgcube=bkgcube, edispcube=edispcube,
                                    caldb=None, irf=None,
                                    edisp=self.spec_edisp,
                                    statistic=self.method_stat,
                                    components=True,
                                    stack=True,
                                    mask=False,
                                    algorithm='SIGNIFICANCE',
                                    logfile=self.output_dir+'/Ana_Spectrum_Residual_log.txt',
                                    silent=self.silent)
                

    #==================================================
    # Timing analysis
    #==================================================
    
    def run_ana_timing(self, tbinalg='LIN', rad=0.2*u.deg, bkgregmin=2, maxoffset=4.0*u.deg):
        """
        Run the timing analysis. This computes the lightcurve
        of the sources in the model, in particular to check that the cluster 
        emission is steady. In case of mismodeling, and due to degeneracies 
        especially with central galaxies, a non steady emission could arrise.
        
        Parameters
        ----------
        - tbinalg (str): binning algorithm, either LIN (linear) or GTI (good time interval)
        - rad (quantity): ON/OFF param - size of the source aperture in deg
        - bkgregmin (int): ON/OFF param - minimum number of off regions
        - maxoffset (quantity): ON/OFF param - maximum offset from camera center 
        to source allowed

        Outputs files
        -------------
        - Ana_Lightcurve_{*}.fits: lightcurve fits file for the sources in the ROI
        - Ana_Lightcurve_{*}_log.fits: lightcurve log file for the sources in the ROI

        """

        print('===== WARNING: only the ON/OFF lightcurve is available for the moment due to a strange bug =====')

        models = gammalib.GModels(self.output_dir+'/Ana_Model_Output.xml')
        Nsource = len(models)

        #========== Get the time start and stop from the observations
        if self.time_tmin is None:
            tmins = Time(self.obs_setup.tmin, format='isot', scale='utc')
            tmin = Time(np.amin(tmins.mjd), format='mjd', scale='utc').fits
        else:
            tmin = self.time_tmin
            
        if self.time_tmax is None:
            tmaxs = Time(self.obs_setup.tmax, format='isot', scale='utc')
            tmax = Time(np.amax(tmaxs.mjd), format='mjd', scale='utc').fits
        else:
            tmax = self.time_tmax
            
        #========== Loop over the sources
        for isource in range(Nsource):

            srcname = models[isource].name()

            if models[isource].type() not in ['CTACubeBackground', 'CTAIrfBackground']:
                if not self.silent:
                    print('--- Computing lightcurve: '+srcname)

                    #----- Define ref coordinates (for on/off)
                    if srcname == self.cluster.name:
                        xref = self.cluster.coord.ra.to_value('deg')
                        yref = self.cluster.coord.dec.to_value('deg')
                    else:
                        xref = models[isource].spatial().ra()
                        yref = models[isource].spatial().dec()

                    #----- Compute lightcurve
                    tools_timing.lightcurve(self.output_dir+'/Ana_EventsSelected.xml', #
                                            self.output_dir+'/Ana_Model_Output.xml',   #
                                            srcname, self.output_dir+'/Ana_Lightcurve_'+srcname+'.fits', #
                                            self.map_reso, self.map_coord, self.map_fov, #
                                            xref=xref,
                                            yref=yref,
                                            caldb=None,
                                            irf=None,
                                            inexclusion=None,
                                            edisp=self.spec_edisp,
                                            tbinalg=tbinalg,
                                            tmin=tmin,
                                            tmax=tmax,
                                            mjdref=51544.5,
                                            tbins=self.time_nbin,
                                            tbinfile=None,
                                            method='ONOFF',
                                            emin=self.spec_emin.to_value('TeV'),
                                            emax=self.spec_emax.to_value('TeV'),
                                            enumbins=self.spec_enumbins,
                                            # For ON/OFF
                                            rad=rad.to_value('deg'),
                                            bkgregmin=bkgregmin,
                                            use_model_bkg=False,
                                            maxoffset=maxoffset.to_value('deg'),
                                            etruemin=self.spec_emin.to_value('TeV'),
                                            etruemax=self.spec_emax.to_value('TeV'),
                                            etruebins=self.spec_enumbins,
                                            #
                                            statistic=self.method_stat,
                                            calc_ts=True,
                                            calc_ulim=True,
                                            fix_srcs=True,
                                            fix_bkg=False,
                                            logfile=self.output_dir+'/Ana_Lightcurve_'+srcname+'_log.txt',
                                            silent=self.silent)

        
    #==================================================
    # Compute expected model 
    #==================================================
    
    def run_ana_expected_output(self, profile_reso=0.05*u.deg):
        """
        Run the expected output, i.e. compute the model given the input cluster 
        in the simulation, if any.
        
        Parameters
        ----------
        - profile_reso (quantity): the resolutioin of the profile

        Outputs files
        -------------
        - Ana_Expected_Cluster.xml: xml model file of the excpected signal
        - Ana_Expected_Cluster_Counts_log.txt: log file for counts cube
        - Ana_Expected_Cluster_Counts.fits: the count cube expected model
        - Ana_Expected_Cluster_profile.fits: the profile centered on the cluster coordinates
        
        Outputs files
        -------------
        - Ana_Expected_Cluster.xml: expected cluster signal model xml file
        - Ana_Expected_Cluster_Counts.fits: expected cluster model counts
        - Ana_Expected_Cluster_Counts_log.txt: expected cluster model counts logfile
        - Ana_Expected_Cluster_profile.fits: counts profile for the expected model

        """
        
        try:
            #----- Make sure the map definition is ok
            if self.map_UsePtgRef:
                self._match_cluster_to_pointing()
                self._match_anamap_to_pointing()
                
            #----- Compute the expected spectrum
            model_exp = gammalib.GModels()
            build_ctools_model.cluster(model_exp,
                                       self.output_dir+'/Sim_Model_Map.fits',
                                       self.output_dir+'/Sim_Model_Spectrum.txt',
                                       ClusterName=self.cluster.name)
            model_exp.save(self.output_dir+'/Ana_Expected_Cluster.xml')
            
            #----- Compute the expected count map
            modcube_Cl = cubemaking.model_cube(self.output_dir,
                                               self.map_reso, self.map_coord, self.map_fov,
                                               self.spec_emin, self.spec_emax,
                                               self.spec_enumbins, self.spec_ebinalg,
                                               edisp=self.spec_edisp,
                                               stack=True, silent=self.silent,
                                               logfile=self.output_dir+'/Ana_Expected_Cluster_Counts_log.txt',
                                               inmodel_usr=self.output_dir+'/Ana_Expected_Cluster.xml',
                                               outmap_usr=self.output_dir+'/Ana_Expected_Cluster_Counts.fits')
            
            #----- Extract the profile
            mcube = fits.open(self.output_dir+'/Ana_Expected_Cluster_Counts.fits')
            header = mcube[0].header
            model_cnt_map = np.sum(mcube[0].data, axis=0)
            header.remove('NAXIS3')
            header['NAXIS'] = 2
            
            r_mod, p_mod, err_mod = radial_profile_cts(model_cnt_map,
                                                       [self.cluster.coord.icrs.ra.to_value('deg'),
                                                        self.cluster.coord.icrs.dec.to_value('deg')],
                                                       stddev=np.sqrt(model_cnt_map), header=header,
                                                       binsize=profile_reso.to_value('deg'),
                                                       stat='POISSON',
                                                       counts2brightness=True)
            tab  = Table()
            tab['radius']  = Column(r_mod, unit='deg',
                                    description='Cluster offset (bin='+str(profile_reso.to_value('deg'))+'deg')
            tab['profile'] = Column(p_mod, unit='deg-2', description='Counts per deg-2')
            tab['error']   = Column(err_mod, unit='deg-2', description='Counts per deg-2 uncertainty')
            tab.write(self.output_dir+'/Ana_Expected_Cluster_profile.fits', overwrite=True)
            
        except:
            print('----- Cannot build the expected model.')
            print('      Maybe the cluster used in the simulation is null.')
            print('      Maybe the cluster used in the simulation is null.')
            

    #==================================================
    # Compute MCMC a posteriori constraints
    #==================================================
    
    def run_ana_mcmc(self,
                     do_spectrum=True, do_profile=True,
                     reset_mcmc=True, run_mcmc=True):
        """
        Fit the spectrum and profile a posteriori
        
        Parameters
        ----------
        do_spectrum (bool): run the MCMC analysis on the spectrum?
        do_profile (bool): run the MCMC analysis on the profile?
        reset_mcmc (bool): reset the existing MCMC chains?
        run_mcmc (bool): run the MCMC sampling?

        Outputs files
        -------------
        - Ana_ResmapCluster_profile_MCMC_sampler.pkl: MCMC profile emcee sampler
        - Ana_ResmapCluster_profile_MCMC_chainstat.txt: chain statistics for the profile
        - Ana_spectrum_Perseus_MCMC_sampler.pkl: MCMC spectrum emcee sampler
        - Ana_spectrum_Perseus_MCMC_chainstat.txt: chain statistics for the spectrum
        
        """

        #----- Spectrum
        if do_spectrum:
            spectrum_file = self.output_dir+'/Ana_Spectrum_'+self.cluster.name+'.fits'
            cluster_test = copy.deepcopy(self.cluster)
            cluster_test.output_dir = self.output_dir

            try:
                mcmc_spectrum.run_spectrum_constraint(cluster_test,
                                                      spectrum_file,
                                                      nwalkers=self.mcmc_nwalkers,
                                                      nsteps=self.mcmc_nsteps,
                                                      burnin=self.mcmc_burnin,
                                                      conf=self.mcmc_conf,
                                                      Nmc=self.mcmc_Nmc,
                                                      reset_mcmc=reset_mcmc,
                                                      run_mcmc=run_mcmc,
                                                      Emin=self.spec_emin.to_value('GeV'),
                                                      Emax=self.spec_emax.to_value('GeV'))
            except FileNotFoundError:
                print(self.output_dir+'/Ana_Spectrum_'+self.cluster.name+'.fits not found, no MCMC')
                
        #----- Profile
        if do_profile:
            profile_files = [self.output_dir+'/Ana_ResmapCluster_profile.fits',
                             self.output_dir+'/Ana_Expected_Cluster_profile.fits']
            cluster_test  = copy.deepcopy(self.cluster)
            cluster_test.output_dir = self.output_dir

            try:
                mcmc_profile.run_profile_constraint(cluster_test,
                                                    profile_files,
                                                    nwalkers=self.mcmc_nwalkers,
                                                    nsteps=self.mcmc_nsteps,
                                                    burnin=self.mcmc_burnin,
                                                    conf=self.mcmc_conf,
                                                    Nmc=self.mcmc_Nmc,
                                                    reset_mcmc=reset_mcmc,
                                                    run_mcmc=run_mcmc)
            except FileNotFoundError:
                print(self.output_dir+'/Ana_ResmapCluster_profile.fits not found, no MCMC')
                
            
    #==================================================
    # Run the plotting tools
    #==================================================
    
    def run_ana_plot(self, obsID=None,
                     smoothing_FWHM=0.1*u.deg,
                     profile_log=True):
        """
        Run the plot analysis
        
        Parameters
        ----------
        - obsID (str): list of obsID to be used in data preparation. 
        By default, all of the are used.
        - smoothing_FWHM (quantity): the smoothing used for skymaps
        - profile_log (bool): show the profile in log scale
        
        Outputs files
        -------------
        - Many plots are obtained from the available file products

        """
        
        #========== Get the obs ID to run (defaults is all of them)
        obsID = self._check_obsID(obsID)
        if not self.silent:
            print('----- ObsID to be looked at: '+str(obsID))
            print('')

        #========== Plot the observing properties
        clustpipe_ana_plot.observing_setup(self)
     
        #========== Show events
        clustpipe_ana_plot.events_quicklook(self, obsID, smoothing_FWHM=smoothing_FWHM)
        
        #========== Show Combined map
        clustpipe_ana_plot.combined_maps(self, obsID, smoothing_FWHM=smoothing_FWHM)
     
        #========== Profile plot
        clustpipe_ana_plot.profile(self, profile_log=profile_log)
    
        #========== Spectrum
        clustpipe_ana_plot.spectrum(self)
        
        #========== Lightcurve
        clustpipe_ana_plot.lightcurve(self)

        #========== Parameter fit correlation matrix
        clustpipe_ana_plot.covmat(self)
        
