"""
This file contains the CTAana class. It is dedicated to the construction of a 
CTAana object, which defines how the CTA analysis of a cluster would proceed.

"""

#==================================================
# Requested imports
#==================================================

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord


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

        #----- Data preparation
        self.data_preparation()
        
        #----- Likelihood analysis
        self.likelihood_analysis()

        #----- Imaging analysis
        self.timing_analysis()
         
        #----- Spectral analysis
        self.spectral_analysis()

        #----- Profile analysis
        self.profile_analysis()

        #----- Imaging analysis
        self.imaging_analysis()

        #----- Output plots
        self.plots()

        
    #==================================================
    # Data preparation
    #==================================================
    
    def run_prep_data(self):
        """
        This fucntion is used to prepare the data to the 
        analysis.
        
        Parameters
        ----------
        
        """

    #==================================================
    # Run the likelihood analysis
    #==================================================
    
    def run_ana_likelihood(self):
        """
        Run the likelihood analysis
        
        Parameters
        ----------
        
        """


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

        
        





