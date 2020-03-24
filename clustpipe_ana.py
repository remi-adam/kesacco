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
import copy
import ctools
import gammalib

from ClusterPipe.Tools import tools_spectral
from ClusterPipe.Tools import tools_imaging
from ClusterPipe.Tools import tools_timing
from ClusterPipe.Tools import plotting
from ClusterPipe.Tools import cubemaking
from ClusterPipe.Tools import utilities
from ClusterPipe.Tools import build_ctools_model
from ClusterPipe       import clustpipe_ana_plot

from ClusterModel.ClusterTools.map_tools import radial_profile


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
                     profile_log=True):
        """
        Run the standard cluster analysis.
        
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
        #self.run_ana_timing()

        #----- Expected output computation
        self.run_ana_expected_output(obsID,
                                     profile_reso=profile_reso)
        
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
            
            radius, prof, err = radial_profile(res_counts,
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
    # Compute expected model 
    #==================================================
    
    def run_ana_expected_output(self, obsID,
                                profile_reso=0.05*u.deg):
        """
        Run the expected output
        
        Parameters
        ----------
        
        """

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
        
        r_mod, p_mod, err_mod = radial_profile(model_cnt_map,
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
        
            
    #==================================================
    # Run the plotting tools
    #==================================================
    
    def run_ana_plot(self, obsID=None,
                     smoothing_FWHM=0.1*u.deg,
                     profile_log=True):
        """
        Run the plots
        
        Parameters
        ----------
        
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
    
        #========== Parameter fit correlation matrix
        clustpipe_ana_plot.covmat(self)
