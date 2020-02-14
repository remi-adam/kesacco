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
    # 
    #==================================================
    
    def run_analysis(self):
        """
        Run the standard cluster analysis.
        
        Parameters
        ----------
        
        """

        
    #==================================================
    # Data preparation
    #==================================================
    
    def run_ana_dataprep(self, obsID=None, UsePtgRef=True):
        """
        This fucntion is used to prepare the data to the 
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

        #----- Data selection
        ##sel = ctools.ctselect()
        ##sel['inobs']  = self.output_dir+'/AnaObsDef.xml'
        ##sel['outobs'] = self.output_dir+'/AnaObsDefSelected.xml'
        ##sel['prefix'] = ''
        ##sel['rad']    = self.map_fov.to_value('deg')
        ##sel['ra']     = self.map_coord.icrs.ra.to_value('deg')
        ##sel['dec']    = self.map_coord.icrs.dec.to_value('deg')
        ##sel['emin']   = self.spec_emin.to_value('TeV')
        ##sel['emax']   = self.spec_emax.to_value('TeV')
        ##if self.time_tmin is not None:
        ##    sel['tmin'] = self.time_tmin
        ##else:
        ##    sel['tmin'] = 'NONE'
        ##if self.time_tmax is not None:
        ##    sel['tmax'] = self.time_tmax
        ##else:
        ##    sel['tmax'] = 'NONE'
        ##sel.execute()
        ##print(sel)
        
        #----- Model
        self._make_model(prefix='AnaModelInput')

        #----- Binning
        if self.method_binned:
            print('coucou')
            binning(self.output_dir+'/AnaExpCube')
            binning(self.output_dir+'/AnaPsfCube')
            binning(self.output_dir+'/AnaBkgCube')
            binning(self.output_dir+'/AnaEdispCube')
        
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
        like['inobs']    = self.output_dir+'/Events.xml'
        
        # Input model XML file.
        like['inmodel']  = self.output_dir+'/AnaModelInput.xml'
        
        # Input exposure cube file.
        if self.method_binned :
            like['expcube']  = self.output_dir+'/AnaExpCube'
            
        # Input PSF cube file
        if self.method_binned :
            like['psfcube']  = self.output_dir+'/AnaPsfCube'
            
        # Input background cube file.
        if self.method_binned :
            like['bkgcube']  = self.output_dir+'/AnaBkgCube'
            
        # Input energy dispersion cube file.
        if self.method_binned and self.spec_edisp:
            like['edispcube']  = self.output_dir+'/AnaEdispCube'
            
        # Calibration database.
        #like['caldb']  =
        
        # Instrument response function.
        #like['irf']  = 

        # Applies energy dispersion to response computation.
        like['edisp']  = self.spec_edisp

        # Output model XML file with values and uncertainties updated by the maximum likelihood fit.
        like['outmodel'] = self.output_dir+'/AnaModelOutput.xml'

        # Output FITS or CSV file to store covariance matrix.
        like['outcovmat']  = self.output_dir+'/AnaModelCovmat.fits'

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


    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_spectral(self):
        """
        Run the spectral analysis
        
        Parameters
        ----------
        
        """


    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_profile(self):
        """
        Run the profile analysis
        
        Parameters
        ----------
        
        """

        
    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_imaging(self):
        """
        Run the imaging analysis
        
        Parameters
        ----------
        
        """

        
        




