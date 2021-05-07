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
from kesacco.Tools import tools_onoff
from kesacco.Tools import plotting
from kesacco.Tools import cubemaking
from kesacco.Tools import utilities
from kesacco.Tools import build_ctools_model
from kesacco.Tools import make_cluster_template
from kesacco.Tools import mcmc_spectrum
from kesacco.Tools import mcmc_profile
from kesacco.Tools import mcmc_spectralimaging1
from kesacco.Tools import mcmc_spectralimaging2
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
    - run_ana_mcmc_spectrum: MCMC fit of the spectrum, assuming a given spatial model
    and using the spectrum extracted from the spectral analysis
    - run_ana_mcmc_profile: MCMC fit of the profile
    - run_ana_mcmc_spectralimaging: run the MCMC to constrain the spectrum and profile
    together with background parameters
    - run_ana_plot: run the plotting tools to show the results

    To do list
    ----------
    
    """
    
    #==================================================
    # Run the baseline data analysis
    #==================================================
    
    def run_analysis(self,
                     obsID=None,
                     do_like=True,
                     do_upperlimit=False,
                     do_img=True,
                     do_spec=True,
                     do_timing=False,
                     do_expected_output=True,
                     do_plot=True):
        """
        Run the standard cluster analysis. This pipeline runs the main
        sub-modules one by one and provides the results in the end. Some
        sub-modules can be switched on/off via the keyword arguments.
        
        Parameters
        ----------
        - obsID (str or str list): list of runs to be observed
        - do_like (bool): do the likelihood fit
        - do_upperlimit (bool): do the Cluster upper limit calculation
        - do_img (bool): do the imaging analysis
        - do_spec (bool): do the spectral analysis
        - do_timing (bool): do the timing analysis
        - do_expected_output (bool): compute the expected outputs 
        according to the known input simulation
        - do_plot (bool): make the plots
        
        """

        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False

        #----- Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster template defined according to pointings
            self._match_anamap_to_pointing()       # Analysis map defined usingaccording to pointings
        
        #----- Data preparation (mandatory)
        dataprep = self.run_ana_dataprep(obsID=obsID)
        
        #----- Likelihood fit
        if do_like:
            like = self.run_ana_likelihood()
        
        #----- Upper limit
        if do_upperlimit:
            UpLim = self.run_ana_upperlimit()
        
        #----- Imaging analysis
        if do_img:
            self.run_ana_imaging()
        
        #----- Spectral analysis
        if do_spec:
            self.run_ana_spectral()

        #----- Timing analysis
        if do_timing:
            self.run_ana_timing()

        #----- Expected output computation
        if do_expected_output:
            self.run_ana_expected_output()
        
        #----- Output plots
        if do_plot:
            self.run_ana_plot(obsID=obsID,
                              smoothing_FWHM=0.1*u.deg,
                              profile_log=True)
        
        
    #==================================================
    # Data preparation
    #==================================================
    
    def run_ana_dataprep(self,
                         obsID=None,
                         frac_src_on_reg=0.8,
                         exclu_rad=0.2*u.deg,
                         use_model_bkg=False,
                         overwrite_data=True,
                         overwrite_irfs=True,
                         PSFmapreso=None):
        """
        This function is used to prepare the data to the 
        analysis.
        
        Parameters
        ----------
        - obsID (str): list of obsID to be used in data preparation. 
        By default, all of the are used.
        - frac_src_on_reg (float): fraction of source flux in the on region,
        used to define the OnOff analysis
        - exclu_rad (quantity): exclusion radius for sources in the field of 
        view in onoff analysis
        - use_model_bkg (bool): do we use background model in on off analysis
        - overwrite_{sel,} (bool): recompute and overwrite existing file, or use them 
        without recomputing
        - PSFmapreso (float, quantity): in the case of large data, the PSF map resolution 
        can be reduced. ctools encourage to use 1.0 deg. By default, we use the standard 
        value but this may produce very large files for small bining and large FoV

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

        Outputs
        -------
        - tuple containing: model_tot, ctscube_stack, ctscube_unstack, expcube, 
        psfcube, bkgcube, and edcube, depending on the requested analysis

        """

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('            Starting the data preparation             ')
            print('======================================================')
            print('')
        
        #----- Init the output list
        outs = []
        
        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False

        #----- Check onoff/Edisp
        if self.method_ana == 'ONOFF' and self.spec_edisp == False:
            print('WARNING: The ONOFF method always uses energy dispersion.') 
            print('         This should be used when simulating the data.  ')
                
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
                                            obsID, self.obs_setup,
                                            updateIRF=True)

        #----- Data selection
        selexist = os.path.exists(self.output_dir+'/Ana_EventsSelected.xml')
        if not overwrite_data and selexist:
            if not self.silent:
                print('-----> Skipping data selection')
        else:
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
            sel['usethres'] = 'USER'
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
            
        model_tot = self._make_model(prefix='Ana_Model_Input',
                                     obsID=obsID) # Compute the model files
        outs.append(model_tot)
        
        #----- Binning (needed even if unbinned likelihood)
        # Data counts
        ctscube1exist = os.path.exists(self.output_dir+'/Ana_Countscube.fits')
        ctscube2exist = os.path.exists(self.output_dir+'/Ana_Countscube.xml')
        if not overwrite_data and ctscube1exist and ctscube1exist:
            if not self.silent:
                print('-----> Skipping data cubemaking')
        else:
            for stacklist in [True, False]: # 1 single fits for stacked, else xml +N fits
                ctscube = cubemaking.counts_cube(self.output_dir,
                                                 self.map_reso, self.map_coord, self.map_fov,
                                                 self.spec_emin, self.spec_emax,
                                                 self.spec_enumbins, self.spec_ebinalg,
                                                 stack=stacklist,
                                                 logfile=self.output_dir+'/Ana_Countscube_log.txt',
                                                 silent=self.silent)
                outs.append(ctscube)

        # Exposure
        expcubeexist = os.path.exists(self.output_dir+'/Ana_Expcube.fits')
        if not overwrite_irfs and expcubeexist:
            if not self.silent:
                print('-----> Skipping exposure cubemaking')
        else:
            expcube = cubemaking.exp_cube(self.output_dir,
                                          self.map_reso, self.map_coord, self.map_fov,
                                          self.spec_emin, self.spec_emax,
                                          self.spec_enumbins, self.spec_ebinalg,
                                          logfile=self.output_dir+'/Ana_Expcube_log.txt',
                                          silent=self.silent)
            outs.append(expcube)

        # PSF
        psfcubeexist = os.path.exists(self.output_dir+'/Ana_Psfcube.fits')
        if not overwrite_irfs and psfcubeexist:
            if not self.silent:
                print('-----> Skipping PSF cubemaking')
        else:
            if PSFmapreso is not None:
                map_reso_psf = PSFmapreso
                useincube = False
            else:
                map_reso_psf = self.map_reso
                useincube = True
            psfcube = cubemaking.psf_cube(self.output_dir,
                                          map_reso_psf, self.map_coord, self.map_fov,
                                          self.spec_emin, self.spec_emax,
                                          self.spec_enumbins, self.spec_ebinalg,
                                          logfile=self.output_dir+'/Ana_Psfcube_log.txt',
                                          silent=self.silent, useincube=useincube)
            outs.append(psfcube)
            
        # Background
        bkgcubeexist1 = os.path.exists(self.output_dir+'/Ana_Bkgcube.fits')
        bkgcubeexist2 = os.path.exists(self.output_dir+'/Ana_Model_Input_Stack.xml')
        if not overwrite_irfs and bkgcubeexist1 and bkgcubeexist2:
            if not self.silent:
                print('-----> Skipping background cubemaking')
        else:
            bkgcube = cubemaking.bkg_cube(self.output_dir,
                                          logfile=self.output_dir+'/Ana_Bkgcube_log.txt',
                                          silent=self.silent)
            outs.append(bkgcube)

        # Energy dispersion
        if self.spec_edisp:
            edispcubeexist = os.path.exists(self.output_dir+'/Ana_Edispcube.fits')
            if not overwrite_irfs and edispcubeexist:
                if not self.silent:
                    print('-----> Skipping Edisp cubemaking')
            else:
                edcube = cubemaking.edisp_cube(self.output_dir,
                                               self.map_coord, self.map_fov,
                                               self.spec_emin, self.spec_emax,
                                               self.spec_enumbins, self.spec_ebinalg,
                                               logfile=self.output_dir+'/Ana_Edispcube_log.txt',
                                               silent=self.silent)
                outs.append(edcube)
        
        #----- ON/OFF files
        if self.method_ana == 'ONOFF':
            # Get the radius to have frac_src_on_reg * flux tot in the on region
            rad = tools_onoff.containment_on_source_fraction(self.cluster, frac_src_on_reg,
                                                             self.spec_emin, self.spec_emax)
            # Compute an exclusion map to avoid other sources
            exclumap = tools_onoff.build_exclusion_map(self.compact_source,
                                                       self.map_coord, self.map_reso, self.map_fov,
                                                       self.output_dir+'/Ana_OnOff_exclusion.fits',
                                                       rad=exclu_rad)
            # Make a source model for On/Off
            tools_onoff.make_onoff_source_model(self.output_dir+'/Ana_Model_Input_Unstack.xml',
                                                self.output_dir+'/Ana_Model_Input_OnOffIn.xml',
                                                self.compact_source, keepbkg=use_model_bkg)
            
            # Run the onoff data preparation
            if self.method_stack:
                obsdefonoff = self.output_dir+'/Ana_ObsDef_OnOff_Stack.xml'
                inmodelonoff = self.output_dir+'/Ana_Model_Input_OnOff_Stack.xml'
            else:
                obsdefonoff = self.output_dir+'/Ana_ObsDef_OnOff_Unstack.xml'
                inmodelonoff = self.output_dir+'/Ana_Model_Input_OnOff_Unstack.xml'


            # region gen
            onoffexist = os.path.exists(self.output_dir+'/Ana_OnOff_on.reg')
            if not overwrite_data and onoffexist:
                if not self.silent:
                    print('-----> Skipping data cubemaking')
            else:
                onoff = tools_onoff.onoff_filegen(self.output_dir+'/Ana_EventsSelected.xml',
                                                  self.output_dir+'/Ana_Model_Input_OnOffIn.xml',
                                                  self.cluster.name, 
                                                  obsdefonoff, inmodelonoff,
                                                  self.output_dir+'/Ana_OnOff',
                                                  self.spec_ebinalg,
                                                  self.spec_emin.to_value('TeV'),
                                                  self.spec_emax.to_value('TeV'),
                                                  self.spec_enumbins,
                                                  self.cluster.coord.ra.to_value('deg'),
                                                  self.cluster.coord.dec.to_value('deg'),
                                                  rad.to_value('deg'),
                                                  inexclusion=self.output_dir+'/Ana_OnOff_exclusion.fits',
                                                  bkgregmin=2, bkgregskip=1,
                                                  use_model_bkg=use_model_bkg, maxoffset=4.0,
                                                  stack=self.method_stack,
                                                  etruemin=0.01,etruemax=300,etruebins=30,
                                                  logfile=self.output_dir+'/Ana_OnOff_log.txt')
            
            if self.method_stack and use_model_bkg:
                tools_onoff.rename_bkg_onoff(inmodelonoff)
                
            outs.append(onoff)
            
        return tuple(outs) #model_tot,ctscube_stack,ctscube_unstack,expcube,psfcube,bkgcube,edcube,onoff
        
        
    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_likelihood(self, refit=False,
                           like_accuracy=0.005,
                           max_iter=50,
                           fix_spat_for_ts=False,
                           compute_bestfit=True):
        """
        Run the likelihood analysis.
        See http://cta.irap.omp.eu/ctools/users/reference_manual/ctlike.html
        
        Parameters
        ----------
        - refit (bool): Perform refitting of solution after initial fit.
        - like_accuracy (float): Absolute accuracy of maximum likelihood value
        - max_iter (int): Maximum number of fit iterations.
        - fix_spat_for_ts (bool): Fix spatial parameters for TS computation.
        - compute_bestfit (bool): compute the best fit model

        Outputs files
        -------------
        - Ana_Model_Output.xml: constrained sky model
        - Ana_Model_Output_log.txt: log file of the likelihood fit
        - Ana_Model_Output_Covmat.fits: covariance matrix for the output fit
        - Ana_Model_Output_Cluster.xml: constrained sky model excluding the cluster
        - Ana_Model_Cube.fits: Best fit model cube for the all data
        - Ana_Model_Cube_log.txt: log file of the best fit model cube for the all data
        - Ana_Model_Cube_Cluster.fits: Best fit model cube for the all data without the cluster
        - Ana_Model_Cube_cluster_log.txt: log file of the best fit model cube for the all data 
        without the cluster
        
        """

        #----- Starting the likelihood fit
        if not self.silent:
            print('')
            print('======================================================')
            print('              Starting the likelihood fit             ')
            print('======================================================')
            print('')
            
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
        if self.method_ana == 'ONOFF':
            if not self.silent:
                print('The OnOff likelihood fit may crash if the pointing')
                print('strategy or On/Off region definition is not appropriate')
            if self.method_stack:
                like['inobs'] = self.output_dir+'/Ana_ObsDef_OnOff_Stack.xml'
            else:
                like['inobs'] = self.output_dir+'/Ana_ObsDef_OnOff_Unstack.xml'
        else:
            if self.method_binned:
                if self.method_stack:
                    like['inobs'] = self.output_dir+'/Ana_Countscube.fits'
                else:
                    like['inobs'] = self.output_dir+'/Ana_Countscube.xml'
            else:
                like['inobs']     = self.output_dir+'/Ana_EventsSelected.xml'

        # Input model XML file.
        if self.method_ana == 'ONOFF':
            if self.method_stack:
                like['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Stack.xml'
            else:
                like['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Unstack.xml'
        else:
            if self.method_binned and self.method_stack:
                like['inmodel']  = self.output_dir+'/Ana_Model_Input_Stack.xml'
            else:
                like['inmodel']  = self.output_dir+'/Ana_Model_Input_Unstack.xml'

        if self.method_ana != 'ONOFF':
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
        else:
            like['expcube']    = 'NONE'
            like['psfcube']    = 'NONE'
            like['bkgcube']    = 'NONE'
            like['edispcube']  = 'NONE'
                
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
        
        if not self.silent:
            print(like)
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
        if compute_bestfit:
            modcube = cubemaking.model_cube(self.output_dir,
                                            self.map_reso, self.map_coord, self.map_fov,
                                            self.spec_emin, self.spec_emax, self.spec_enumbins,
                                            self.spec_ebinalg,
                                            edisp=self.spec_edisp,
                                            stack=self.method_stack,
                                            logfile=self.output_dir+'/Ana_Model_Cube_log.txt',
                                            silent=self.silent)
            
            modcube_Cl = cubemaking.model_cube(self.output_dir,
                                               self.map_reso, self.map_coord, self.map_fov,
                                               self.spec_emin, self.spec_emax, self.spec_enumbins,
                                               self.spec_ebinalg,
                                               edisp=self.spec_edisp,
                                               stack=self.method_stack, silent=self.silent,
                                               logfile=self.output_dir+'/Ana_Model_Cube_Cluster_log.txt',
                                               inmodel_usr=self.output_dir+'/Ana_Model_Output_Cluster.xml',
                                               outmap_usr=self.output_dir+'/Ana_Model_Cube_Cluster.fits')
            
            return (like, modcube, modcube_Cl)
        
        else:
            return like
        
        
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

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('         Starting the sensitivity calculation         ')
            print('======================================================')
            print('')

        #----- Select the source name
        if source_name is None:
            srcname = self.cluster.name
        else:
            srcname = source_name

        #----- number of pixels for bining
        npix = utilities.npix_from_fov_def(self.map_fov, self.map_reso)

        #----- Fill the parameters and run
        sens = cscripts.cssens()

        # Input model XML file.
        if self.method_ana == 'ONOFF':
            if self.method_stack:
                sens['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Stack.xml'
            else:
                sens['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Unstack.xml'
        else:
            if self.method_binned and self.method_stack:
                sens['inmodel']  = self.output_dir+'/Ana_Model_Input_Stack.xml'
            else:
                sens['inmodel']  = self.output_dir+'/Ana_Model_Input_Unstack.xml'

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
        
        return sens

    
    #==================================================
    # Upper limit
    #==================================================
    
    def run_ana_upperlimit(self,
                           CL=0.95,
                           sigma_min=0.0, sigma_max=10.0,
                           like_accuracy=0.005,
                           max_iter=50,
                           eref=1*u.TeV):
        
        """
        Get the upper limit on the fit parameters
        
        Parameters
        ----------
        - CL (float): confidence interval
        - sigma_min/max (flaot): Minimum/Maximum boundary to start searching 
        for upper limit value. Number of standard deviations above best fit value
        - like_accuracy (float): Absolute accuracy of maximum likelihood value
        - max_iter (int): Maximum number of fit iterations.
        - eref (quantity): reference energy for differential flux limit

        Outputs files
        -------------
        
        """

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('         Starting the upper limit calculation         ')
            print('======================================================')
            print('')

        #----- Check binned/stacked
        if self.method_binned == False:
            self.method_stack = False
        
        #----- Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()
            self._match_anamap_to_pointing()
        
        #========== Run the upper limit
        if not self.silent:
            if (not self.method_binned) and self.method_stack:
                print('WARNING: unbinned likelihood are not stacked')
                print('')

        UL = ctools.ctulimit()

        # Input event list, counts cube or observation definition XML file.
        if self.method_ana == 'ONOFF':
            if self.method_stack:
                UL['inobs'] = self.output_dir+'/Ana_ObsDef_OnOff_Stack.xml'
            else:
                UL['inobs'] = self.output_dir+'/Ana_ObsDef_OnOff_Unstack.xml'
        else:
            if self.method_binned:
                if self.method_stack:
                    UL['inobs'] = self.output_dir+'/Ana_Countscube.fits'
                else:
                    UL['inobs'] = self.output_dir+'/Ana_Countscube.xml'
            else:
                UL['inobs']     = self.output_dir+'/Ana_EventsSelected.xml'

        # Input model XML file.
        if self.method_ana == 'ONOFF':
            if self.method_stack:
                UL['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Stack.xml'
            else:
                UL['inmodel'] = self.output_dir+'/Ana_Model_Input_OnOff_Unstack.xml'
        else:
            if self.method_binned and self.method_stack:
                UL['inmodel']  = self.output_dir+'/Ana_Model_Input_Stack.xml'
            else:
                UL['inmodel']  = self.output_dir+'/Ana_Model_Input_Unstack.xml'

        # Name of source model for which the upper flux limit should be computed.
        UL['srcname'] = self.cluster.name

        if self.method_ana != 'ONOFF':
            # Input exposure cube file.
            if self.method_binned and self.method_stack :
                UL['expcube']  = self.output_dir+'/Ana_Expcube.fits'
                
            # Input PSF cube file
            if self.method_binned and self.method_stack :
                UL['psfcube']  = self.output_dir+'/Ana_Psfcube.fits'
                
            # Input background cube file.
            if self.method_binned and self.method_stack :
                UL['bkgcube']  = self.output_dir+'/Ana_Bkgcube.fits'
                
            # Input energy dispersion cube file.
            if self.method_binned and self.method_stack and self.spec_edisp:
                UL['edispcube']  = self.output_dir+'/Ana_Edispcube.fits'
        else:
            UL['expcube']    = 'NONE'
            UL['psfcube']    = 'NONE'
            UL['bkgcube']    = 'NONE'
            UL['edispcube']  = 'NONE'
                
        # Calibration database
        #UL['caldb']  =
        
        # Instrument response function
        #UL['irf']  = 

        # Applies energy dispersion to response computation.
        UL['edisp']  = self.spec_edisp

        # Confidence level of upper limit.
        UL['confidence'] = CL

        # Minimum boundary to start searching for upper limit value. Number of stddev above best fit
        UL['sigma_min'] = sigma_min

        # Maximum boundary to start searching for upper limit value. Number of stddev above best fit
        UL['sigma_max'] = sigma_max

        # Reference energy for differential limit (in TeV).
        UL['eref'] = eref.to_value('TeV')
        
        # Minimum energy for integral flux limit (in TeV).
        UL['emin'] = self.spec_emin.to_value('TeV')

        # Maximum energy for integral flux limit (in TeV).
        UL['emax'] = self.spec_emax.to_value('TeV')

        # Optimization statistic. 
        UL['statistic']  = self.method_stat

        # Absolute accuracy of maximum likelihood value.
        UL['like_accuracy']  = like_accuracy

        # Maximum number of fit iterations.
        UL['max_iter']  = max_iter

        # Log file
        UL['logfile'] = self.output_dir+'/Ana_Cluster_UpLim_log.txt'

        UL.logFileOpen()
        UL.execute()
        UL.logFileClose()
    
        if not self.silent:
            print(UL)
            print('')

        return UL

    
    #==================================================
    # Run the imaging analysis
    #==================================================
    
    def run_ana_imaging(self, bkgsubtract='NONE',
                        do_Skymap=True,
                        do_SourceDet=False,
                        do_Res=True,
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

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('            Starting the imaging analysis             ')
            print('======================================================')
            print('')
        
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
                                          roiradius=0.04,
                                          inradius=self.cluster.theta500.to_value('deg'),
                                          outradius=self.cluster.theta500.to_value('deg')*1.2,
                                          iterations=3, threshold=3,
                                          logfile=self.output_dir+'/Ana_SkymapTot_log.txt',
                                          silent=self.silent)
        
        #========== Search for sources
        if do_SourceDet:
            if os.path.isfile(self.output_dir+'/Ana_SkymapTot.fits'):
                srcmap = tools_imaging.src_detect(self.output_dir+'/Ana_SkymapTot.fits',
                                                  self.output_dir+'/Ana_Sourcedetect.xml',
                                                  self.output_dir+'/Ana_Sourcedetect.reg',
                                                  threshold=5.0, maxsrcs=10, avgrad=1.0,
                                                  corr_rad=0.05, exclrad=0.2,
                                                  logfile=self.output_dir+'/Ana_Sourcedetect_log.txt',
                                                  silent=self.silent)
            else:
                print(self.output_dir+'/Ana_SkymapTot.fits not found, but needed for source detection.')
                print('')
                
        #========== Compute residual (w/wo cluster subtracted)
        if do_Res:
            if self.method_ana == 'ONOFF':
                print('===== WARNING: the implemented ON/OFF analysis focuses on the residual of the cluster.')
                print('               Other sources unaccounted for in the model may bias the residual.')

            #----- Total residual and keeping the cluster
            for alg in ['SIGNIFICANCE', 'SUB', 'SUBDIV', 'SUBDIVSQRT']:
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
                
                resmap_cl = tools_imaging.resmap(self.output_dir+'/Ana_Countscube.fits',
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
            hdul       = fits.open(self.output_dir+'/Ana_ResmapCluster_SUB.fits')
            res_clall    = hdul[0].data
            header     = hdul[0].header
            hdul.close()
            hdul       = fits.open(self.output_dir+'/Ana_ResmapCluster_SUBDIV.fits')
            subdiv_clall = hdul[0].data
            header     = hdul[0].header
            hdul.close()
            clmodel = res_clall/subdiv_clall

            # Residual counts
            radius, prof, err = radial_profile_cts(res_counts,
                                                   [self.cluster.coord.icrs.ra.to_value('deg'),
                                                    self.cluster.coord.icrs.dec.to_value('deg')],
                                                   stddev=np.sqrt(model), header=header,
                                                   binsize=profile_reso.to_value('deg'), stat='POISSON',
                                                   counts2brightness=True)

            # Background counts
            radius, bkgprof, bkgerr = radial_profile_cts(clmodel,
                                                         [self.cluster.coord.icrs.ra.to_value('deg'),
                                                          self.cluster.coord.icrs.dec.to_value('deg')],
                                                         stddev=np.sqrt(clmodel), header=header,
                                                         binsize=profile_reso.to_value('deg'), stat='POISSON',
                                                         counts2brightness=True)

            
            tab  = Table()
            tab['radius']      = Column(radius, unit='deg',
                                        description='Cluster offset (bin='+str(profile_reso.to_value('deg'))+'deg')
            tab['profile']     = Column(prof,   unit='deg-2', description='Counts per deg-2')
            tab['error']       = Column(err,    unit='deg-2', description='Counts per deg-2 uncertainty')
            tab['bkg_profile'] = Column(bkgprof,    unit='deg-2', description='Bkg counts per deg-2')
            tab['bkg_error']   = Column(bkgerr,    unit='deg-2', description='Bkg counts per deg-2 uncertainty')
            tab['radius_min']  = radius - profile_reso.to_value('deg')/2.0
            tab['radius_max']  = radius + profile_reso.to_value('deg')/2.0
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
    
    def run_ana_spectral(self,
                         do_Spec=True,
                         do_Butterfly=True,
                         do_Res=True,
                         use_Lfit_model=True,
                         fix_srcs=False,
                         fix_bkg=True,
                         method='SLICE',
                         name=None):
        """
        Run the spectral analysis
        
        Parameters
        ----------
        - do_Spec (bool): compute spectra
        - do_Butterfly (bool): compute butterfly
        - do_Res (bool): compute residual spectra
        - use_Lfit_mode (bool): use the model obtained from likelihood fit
        - fix_srcs (bool): fix the point sources in the fit
        - fix_bkg (bool): fix the background in the fit
        - method (str): metgod for computing the spectrum (SLICE/NODES)
        - name (str): name of the source to consider (default all source used)
        
        Outputs files
        -------------
        - Ana_Spectrum_{*}.fits: spectrum file for the sources listed in the ROI
        - Ana_Spectrum_{*}_log.txt: spectrum logfile for the sources listed in the ROI
        - Ana_Spectrum_Buterfly_{*}.txt: butterfly file for the sources listed in the ROI
        - Ana_Spectrum_Buterfly_{*}_log.txt: butterfly log file for the sources listed in the ROI
        - Ana_Spectrum_Residual.fits: residual spectrum
        - Ana_Spectrum_Residual_log.txt: residual spectrum logfile
        
        """

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('            Starting the spectral analysis            ')
            print('======================================================')
            print('')

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

            # Condition for doing specrum
            cond1 = models[isource].type() not in ['CTACubeBackground', 'CTAIrfBackground']
            if name is not None:
                if name == srcname:
                    cond2 = True
                else:
                    cond2 = False
            else:
                cond2 = True
            
            if cond1 and cond2:
                if not self.silent:
                    print('--- Computing spectrum: '+srcname)

                #----- Compute spectra
                if do_Spec:
                    if srcname == self.cluster.name:
                        dll_sigstep = 0.5
                        dll_sigmax  = 7.0
                    else:
                        dll_sigstep = 0.0
                        dll_sigmax  = 5.0

                    if use_Lfit_model:
                        model_file = self.output_dir+'/Ana_Model_Output.xml'
                    else:
                        model_file = inmodel
        
                    tools_spectral.spectrum(inobs,
                                            model_file,
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
                                            method=method, # No impact seen except NODES more robust (no bad pt)
                                            statistic=self.method_stat,
                                            calc_ts=True,
                                            calc_ulim=True,
                                            fix_srcs=fix_srcs,
                                            fix_bkg=fix_bkg,
                                            dll_sigstep=dll_sigstep,
                                            dll_sigmax=dll_sigmax,
                                            dll_freenodes=False,# Errors with True...
                                            logfile=self.output_dir+'/Ana_Spectrum_'+srcname+'_log.txt',
                                            silent=self.silent)

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
            tools_spectral.residual(inobs, model_file,
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

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('             Starting the timing analysis             ')
            print('======================================================')
            print('')

            if self.method_ana != 'ONOFF':
                print('===== WARNING: only ON/OFF lightcurve available for the moment =====')

        #----- Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()
            self._match_anamap_to_pointing()
        
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

        #========== Defines cubes
        inobs, inmodel, expcube, psfcube, bkgcube, edispcube, modcube, modcubeCl = self._define_std_filenames()
            
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
                    tools_timing.lightcurve(self.output_dir+'/Ana_EventsSelected.xml',
                                            self.output_dir+'/Ana_Model_Output.xml', 
                                            srcname, self.output_dir+'/Ana_Lightcurve_'+srcname+'.fits',
                                            self.map_reso, self.map_coord, self.map_fov,
                                            xref=xref,
                                            yref=yref,
                                            caldb=None,
                                            irf=None,
                                            inexclusion=None,
                                            edisp=self.spec_edisp,
                                            tbinalg=tbinalg,
                                            tmin=tmin,
                                            tmax=tmax,
                                            mjdref=self.time_mjdref,
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
                                            etruemin=0.01,
                                            etruemax=300,
                                            etruebins=30,
                                            #
                                            statistic=self.method_stat,
                                            calc_ts=True,
                                            calc_ulim=True,
                                            fix_srcs=True,
                                            fix_bkg=False,
                                            logfile=self.output_dir+'/Ana_Lightcurve_'+srcname+'_log.txt',
                                            silent=self.silent)

        
    #==================================================
    # Compute expected outputs according to input sim
    #==================================================
    
    def run_ana_expected_output(self,
                                profile_reso=0.05*u.deg):
        """
        Compute the expected profile (i.e. irf convolved) according to
        the input simulation model.
        
        Parameters
        ----------
        - profile_reso (quantity): the resolution of the profile

        Outputs files
        -------------
        - Ana_Expected_Cluster.xml: expected cluster signal model xml file
        - Ana_Expected_Cluster_Counts.fits: expected cluster model counts
        - Ana_Expected_Cluster_Counts_log.txt: expected cluster model counts logfile
        - Ana_Expected_Cluster_profile.fits: counts profile for the expected model

        """

        #===== Information
        if not self.silent:
            print('')
            print('======================================================')
            print(' Starting the calculation of the expected output model')
            print('======================================================')
            print('')

        #===== Make sure the map definition is ok
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()
            self._match_anamap_to_pointing()
            
        #===== Compute the expected profile from the simulation
        try:                
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
            print('----- Error in building the expected model.')
            print('      Maybe the cluster used in the simulation is null.')


    #==================================================
    # Compute MCMC spectrum a posteriori constraints
    #==================================================
    
    def run_ana_mcmc_spectrum(self,
                              reset_mcmc=True,
                              run_mcmc=True,
                              GaussLike=False):
        """
        Fit the spectrum a posteriori using a MCMC approach. The 
        profile is kept fixed here.
        
        Parameters
        ----------
        - reset_mcmc (bool): reset the existing MCMC chains?
        - run_mcmc (bool): run the MCMC sampling?
        - GaussLike (bool): use gaussian likelihood instead of 
        the true likelihood scan
        
        Outputs files
        -------------
        - Ana_MCMC_spectrum_*: MCMC spectrum results
        
        """

        #----- Information
        if not self.silent:
            print('')
            print('======================================================')
            print('         Starting the MCMC spectrum analysis          ')
            print('======================================================')
            print('')
            
        spectrum_file = self.output_dir+'/Ana_Spectrum_'+self.cluster.name+'.fits'
        cluster_test = copy.deepcopy(self.cluster)
        cluster_test.output_dir = self.output_dir
        
        if os.path.exists(spectrum_file):
            mcmc_spectrum.run_constraint(cluster_test,
                                         spectrum_file,
                                         nwalkers=self.mcmc_nwalkers,
                                         nsteps=self.mcmc_nsteps,
                                         burnin=self.mcmc_burnin,
                                         conf=self.mcmc_conf,
                                         Nmc=self.mcmc_Nmc,
                                         GaussLike=GaussLike,
                                         reset_mcmc=reset_mcmc,
                                         run_mcmc=run_mcmc,
                                         Emin=self.spec_emin.to_value('GeV'),
                                         Emax=self.spec_emax.to_value('GeV'))
        else:
            print(spectrum_file+' was not found, but is requiered.')
            print('run_ana_spectral should be run first in order to generate this file.')


    #==================================================
    # Compute MCMC profile a posteriori constraints
    #==================================================
    
    def run_ana_mcmc_profile(self,
                             reset_modelgrid=True,
                             reset_mcmc=True,
                             run_mcmc=True,
                             GaussLike=False,
                             profile_reso=0.05*u.deg,
                             spatial_range=[0.0,2.0],
                             spatial_npt=11,
                             includeIC=False,
                             rm_tmp=False):
        """
        Fit the profile a posteriori using a MCMC approach. The 
        spectrum is kept fixed here.
        
        Parameters
        ----------
        - reset_mcmc (bool): reset the existing MCMC chains?
        - run_mcmc (bool): run the MCMC sampling?
        - GaussLike (bool): use gaussian likelihood instead of 
        the true likelihood scan
        
        Outputs files
        -------------
        - Ana_MCMC_profile_*: MCMC spectrum results
        
        """

        #===== Information
        if not self.silent:
            print('')
            print('======================================================')
            print(' Starting the MCMC profile analysis')
            print('======================================================')
            print('')

        #===== Create the subdirectory
        subdir = self.output_dir+'/Ana_MCMC_profile'
        if not os.path.exists(subdir): os.mkdir(subdir)

        #===== Param checks
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings

        if self.method_stack is not True:
            self.method_stack  = True
            print('run_ana_profile_mcmc requires method_stack=True. --> Change applied')
        if self.method_binned is not True:
            self.method_binned  = True
            print('run_ana_profile_mcmc requires method_binned=True. --> Change applied')
        if self.method_ana != '3D':
            self.method_ana  = '3D'
            print('run_ana_profile_mcmc requires method_ana="3D". --> Change applied')            
        
        #===== Build the model grid
        #----- Get the initial CRp profile
        rad      = np.logspace(-1,5,10000)*u.kpc
        prof_ini = self.cluster._get_generic_profile(rad, self.cluster.density_crp_model)
        
        if np.nanmax(prof_ini) == np.nanmin(prof_ini):
            print('----- The input CRp profile model is flat.')
            print('----- Thus, rescaling the model as profile^eta does not change it.')
            print('----- Change the CRp profile model so that it can be used to sample different shapes.')
            raise ValueError('CRp profile model error')

        #----- Define the sampling values
        spatial_value = np.linspace(spatial_range[0], spatial_range[1], spatial_npt)
        spatial_idx   = np.linspace(0, spatial_npt-1, spatial_npt, dtype=np.int)

        #----- Check that parameters are fine
        if reset_modelgrid is False:
            if not os.path.exists(subdir+'/Grid_Parameters.npy'):
                raise ValueError('reset_modelgrid is False, but no previous run was found.')
            
            listpar = np.load(subdir+'/Grid_Parameters.npy', allow_pickle=True)
            cluster_previous = listpar[0]
            spatial_value_previous = listpar[1]
            
            prof_previous = cluster_previous._get_generic_profile(rad, cluster_previous.density_crp_model)
            if not (prof_previous.value == prof_ini.value).all():
                raise ValueError('reset_modelgrid=False, but the cluster object has changed since last run')

            if not (spatial_value_previous == spatial_value).all():
                raise ValueError('reset_modelgrid=False, but the spatial_scaling_value has changed since last run')
        
        #----- Run the grid making
        if reset_modelgrid:
            mcmc_profile.build_model_grid(self,
                                          subdir,
                                          rad, prof_ini,
                                          spatial_value, spatial_idx,
                                          profile_reso,
                                          includeIC=includeIC,
                                          rm_tmp=rm_tmp)
                    
        #===== Run the MCMC
        cluster_test  = copy.deepcopy(self.cluster)
        cluster_test.output_dir = self.output_dir
        
        mcmc_profile.run_profile_constraint(subdir+'/Grid_Sampling.fits',
                                            subdir,
                                            nwalkers=self.mcmc_nwalkers,
                                            nsteps=self.mcmc_nsteps,
                                            burnin=self.mcmc_burnin,
                                            conf=self.mcmc_conf,
                                            Nmc=self.mcmc_Nmc,
                                            GaussLike=GaussLike,
                                            reset_mcmc=reset_mcmc,
                                            run_mcmc=run_mcmc)
        
            
    #==================================================
    # Compute SpectroImaging constraints
    #==================================================
    
    def run_ana_mcmc_spectralimaging(self,
                                     reset_modelgrid=True,
                                     reset_mcmc=True, run_mcmc=True,
                                     GaussLike=False,
                                     spatial_range=[0.0,2.0],
                                     spatial_npt=11,
                                     spectral_range=[2.0,3.0],
                                     spectral_npt=11,
                                     includeIC=False,
                                     bkg_marginalize=True,
                                     bkg_spectral_npt=11,
                                     bkg_spectral_range=[-0.5,0.5],
                                     ps_spectral_npt=11,
                                     ps_spectral_range=[-0.5,0.5],
                                     rm_tmp=False,
                                     Ngrid_validation=0,
                                     FWHM=0.1*u.deg,
                                     theta=1.0*u.deg,
                                     coord=None,
                                     profile_reso=0.05*u.deg):
        """
        Perform a spectral-imaging analysis to constrain the cluster 
        spectrum and profile simulteneously.
        Depending on the keyword bkg_marginalize, the background will be 
        fit together in the MCMC, or marginalized over and fixed to 
        the interpolated maximum likelihood value corresponding to 
        the tested cluster model.
        
        Parameters
        ----------
        - reset_modelgrid (bool): recompute the grid of models even if already exist
        - reset_mcmc (bool): reset the existing MCMC chains?
        - run_mcmc (bool): run the MCMC sampling?
        - GaussLike (bool): use guassian likelihood or true L scan
        - spatial_scaling_npt (int): number of point for the scaling parameter
        - spatial_scaling_range (list of min/max): input profile scaling range
        - includeIC (bool): include inverse Compton in the model?
        - bkg_marginalize (bool): if true, each cluster model used when computing the grid
        is fitted to the data (ctools maximum likelihood) and marginalized over in the MCMC
        fit. Otherwise, a grid is also build for the background and fir together with the 
        cluster in the MCMC.
        - bkg_spectral_npt (int): number of point to sample the background spectrum
        - bkg_spectral_range (list of min/max): min and max value to add to the background
        spectrum, i.e. [default+min, default+max]
        - ps_spectral_npt (int): number of point to sample the point source spectrum
        - ps_spectral_range (list of min/max): min and max value to add to the point sources
        spectra, i.e. [default+min, default+max]
        - rm_tmp (bool): remove temporary templates?
        - Ngrid_validation (int): numbre of model to check for interpolation validation
        - FWHM (quantity): size of the FWHM to be used for smoothing (plot)
        - theta (quantity): containment angle for plots
        - coord (SkyCoord): source coordinates for extraction (plot)
        - profile_reso (quantity): bin size for profile (plot)

        Outputs files
        -------------
        - Results are saved in a dedicated subdirectory named Ana_MCMC_SpecImg1
        
        """
        
        #===== Information
        if not self.silent:
            print('')
            print('======================================================')
            print('      Starting the MCMC spectral imaging analysis     ')
            if bkg_marginalize:
                print('      (using background marginalization)          ')
            else:
                print('      (including the background in the grid model)')
            print('                                                      ')
            print('Note that eta is defined as n_CR_fit ~ n_CR_input^eta ')
            print('If you want eta to be as n_CR_fit ~ n_gas^eta, you may')
            print('set your cluster CR model to verify n_CR_input ~ n_gas')
            print('======================================================')
            print('')
            
        #===== Create the subdirectory
        if bkg_marginalize:
            extra = '1'
        else:            
            extra = '2'
        subdir = self.output_dir+'/Ana_MCMC_SpecImg'+extra
        if not os.path.exists(subdir): os.mkdir(subdir)
        
        #===== Param checks
        if self.map_UsePtgRef:
            self._match_cluster_to_pointing()      # Cluster map defined using pointings
            self._match_anamap_to_pointing()       # Analysis map defined using pointings

        if self.method_stack is not True:
            self.method_stack  = True
            print('run_ana_spectralimaging_mcmc requires method_stack=True. --> Change applied')
        if self.method_binned is not True:
            self.method_binned  = True
            print('run_ana_spectralimaging_mcmc requires method_binned=True. --> Change applied')
        if self.method_ana != '3D':
            self.method_ana  = '3D'
            print('run_ana_spectralimaging_mcmc requires method_ana="3D". --> Change applied')            

        # By default the cluster coordinates are used
        if coord is None:
            coord = self.cluster.coord
            
        #===== Get the initial CRp profile
        rad      = np.logspace(-1,5,10000)*u.kpc
        prof_ini = self.cluster._get_generic_profile(rad, self.cluster.density_crp_model)

        if np.nanmax(prof_ini) == np.nanmin(prof_ini):
            print('----- The input CRp profile model is flat.')
            print('----- Thus, rescaling the model as profile^eta does not change it.')
            print('----- Change the CRp profile model so that it can be used to sample different shapes.')
            raise ValueError('CRp profile model error')
        
        #===== Define the sampling values
        spatial_value  = np.linspace(spatial_range[0],spatial_range[1],spatial_npt)
        spatial_idx    = np.linspace(0, spatial_npt-1, spatial_npt, dtype=np.int)
        spectral_value = np.linspace(spectral_range[0],spectral_range[1],spectral_npt)
        spectral_idx   = np.linspace(0, spectral_npt-1, spectral_npt, dtype=np.int)

        bk_spectral_value = np.linspace(bkg_spectral_range[0],bkg_spectral_range[1], bkg_spectral_npt)
        bk_spectral_idx   = np.linspace(0, bkg_spectral_npt-1, bkg_spectral_npt, dtype=np.int)
        ps_spectral_value = np.linspace(ps_spectral_range[0],ps_spectral_range[1],ps_spectral_npt)
        ps_spectral_idx   = np.linspace(0, ps_spectral_npt-1, ps_spectral_npt, dtype=np.int)

        #===== Check that parameters are fine
        if reset_modelgrid is False:
            if not os.path.exists(subdir+'/Grid_Parameters.npy'):
                raise ValueError('reset_modelgrid is False, but no previous run was found.')
            
            listpar = np.load(subdir+'/Grid_Parameters.npy', allow_pickle=True)
            cluster_previous = listpar[0]
            spatial_value_previous = listpar[1]
            spectral_value_previous = listpar[2]

            prof_previous = cluster_previous._get_generic_profile(rad, cluster_previous.density_crp_model)
            if not (prof_previous.value == prof_ini.value).all():
                raise ValueError('reset_modelgrid=False, but the cluster object has changed since last run')

            if not (spatial_value_previous == spatial_value).all():
                raise ValueError('reset_modelgrid=False, but the spatial_scaling_value has changed since last run')

            if not (spectral_value_previous == spectral_value).all():
                raise ValueError('reset_modelgrid=False, but the spectral_slope_value has changed since last run')
        
        #===== Build the model grid
        if reset_modelgrid:
            if bkg_marginalize:
                mcmc_spectralimaging1.build_model_grid(self,
                                                       subdir,
                                                       rad, prof_ini,
                                                       spatial_value, spatial_idx,
                                                       spectral_value, spectral_idx,
                                                       includeIC=includeIC, rm_tmp=rm_tmp)
            else:
                mcmc_spectralimaging2.build_model_grid(self,
                                                       subdir,
                                                       rad, prof_ini,
                                                       spatial_value, spatial_idx,
                                                       spectral_value, spectral_idx,
                                                       bk_spectral_value, bk_spectral_idx,
                                                       ps_spectral_value, ps_spectral_idx,
                                                       includeIC=includeIC, rm_tmp=rm_tmp)
        #===== Validation of the grid interpolation
        if Ngrid_validation>0:
            print('')
            print('----> Running '+str(Ngrid_validation)+' random validation points in the parameters space')
            print('- Spectral values used for the grid:')
            print(spectral_value)
            print('- Spatial values used for the grid:')
            print(spatial_value)
            
            for ipar in range(Ngrid_validation):
                spec_ps = np.random.uniform(ps_spectral_range[0], ps_spectral_range[1])
                param = [1.0,
                         np.random.uniform(spatial_range[0], spatial_range[1]),
                         np.random.uniform(spectral_range[0], spectral_range[1]),
                         1.0,
                         np.random.uniform(bkg_spectral_range[0], bkg_spectral_range[1]),
                         1.0, spec_ps,1.0,spec_ps]
                mcmc_spectralimaging2.validation_model_grid_itpl(param,
                                                                 [self.output_dir+'/Ana_Countscube.fits',
                                                                  subdir+'/Grid_Sampling.fits'],
                                                                 self,
                                                                 subdir,
                                                                 rad, prof_ini,
                                                                 includeIC=includeIC)

        #===== MCMC fit with cluster parameters
        if bkg_marginalize:
            mcmc_spectralimaging1.run_constraint([self.output_dir+'/Ana_Countscube.fits',
                                                  subdir+'/Grid_Sampling.fits'],
                                                 subdir,
                                                 nwalkers=self.mcmc_nwalkers,
                                                 nsteps=self.mcmc_nsteps,
                                                 burnin=self.mcmc_burnin,
                                                 conf=self.mcmc_conf,
                                                 Nmc=self.mcmc_Nmc,
                                                 GaussLike=GaussLike,
                                                 reset_mcmc=reset_mcmc,
                                                 run_mcmc=run_mcmc,
                                                 FWHM=FWHM,
                                                 theta=theta,
                                                 coord=coord,
                                                 profile_reso=profile_reso)
        else:
            mcmc_spectralimaging2.run_constraint([self.output_dir+'/Ana_Countscube.fits',
                                                  subdir+'/Grid_Sampling.fits'],
                                                 subdir,
                                                 nwalkers=self.mcmc_nwalkers,
                                                 nsteps=self.mcmc_nsteps,
                                                 burnin=self.mcmc_burnin,
                                                 conf=self.mcmc_conf,
                                                 Nmc=self.mcmc_Nmc,
                                                 GaussLike=GaussLike,
                                                 reset_mcmc=reset_mcmc,
                                                 run_mcmc=run_mcmc,
                                                 FWHM=FWHM,
                                                 theta=theta,
                                                 coord=coord,
                                                 profile_reso=profile_reso)
            

    #==================================================
    # Run the plotting tools
    #==================================================
    
    def run_ana_plot(self,
                     obsID=None,
                     ShowIndividualPointing=False,
                     bkgsubtract='NONE',
                     smoothing_FWHM=0.1*u.deg,
                     profile_log=True):
        """
        Run the plot analysis
        
        Parameters
        ----------
        - obsID (str): list of obsID to be used in data preparation. 
        By default, all of the are used.
        - ShowIndividualPointing (bool): show the indicidual pointing maps
        - bkgsubtract (string): method for subtracting the background in skymaps quicklooks
        - smoothing_FWHM (quantity): the smoothing used for skymaps
        - profile_log (bool): show the profile in log scale
        
        Outputs files
        -------------
        - Many plots are obtained from the available file products

        """

        #===== Information
        if not self.silent:
            print('')
            print('======================================================')
            print(' Starting the automatic plots')
            print('======================================================')
            print('')
        
        #========== Get the obs ID to run (defaults is all of them)
        obsID = self._check_obsID(obsID)
        if not self.silent:
            print('----- ObsID to be looked at: '+str(obsID))
            print('')
        
        #========== Plot the observing properties
        clustpipe_ana_plot.observing_setup(self)
     
        #========== Show individual pointings
        if ShowIndividualPointing:
            clustpipe_ana_plot.events_quicklook(self,
                                                obsID,
                                                smoothing_FWHM=smoothing_FWHM,
                                                bkgsubtract=bkgsubtract)
        
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

