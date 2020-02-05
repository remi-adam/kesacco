"""
This file contains the Background class. It is dedicated to the construction of a 
Background object, which can be used in CTA simulations or analysis.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u


#==================================================
# Total background class
#==================================================

class Background(object):
    """ 
    Background class. 
    This class defines the Background object. The background is 
    defined by its attribute (name, spatial, spectral).
    
    Attributes
    ----------  
    - name     : the label of the given background 
    - spatial  : the spatial properties of the background
    - spectral : the spectral properties of the background

    Methods
    ----------  
    - set_spatial_std_bkg
    - print_bkg

    """
    
    #==================================================
    # Initialize the background
    #==================================================
    
    def __init__(self):
        """
        Initialize the background object.
        
        Parameters
        ----------
        - name (str): the label of the given background
        - spatial (dictionary): contain coordinates of the background
        - spectral (dictionary): spectral properties of the background
        
        """
        
        self.name     = 'Background'
        self.spatial  = {'type':'CTAIrfBackground'}
        self.spectral = {'type':'PowerLaw',
                         'param':{'Prefactor':{'value':1.0, 'free':True},
                                  'Index':{'value':0, 'free':True},
                                  'PivotEnergy':{'value':1.0*u.TeV, 'free':False}}}
        
        
    #==================================================
    # Set standard CTAIRF spatial background
    #==================================================
    
    def set_spatial_std_bkg(self, bkg_type='CTAIrfBackground'):
        """
        Set the spatial part of the background to standard model.
        Available standard models are
        - CTAIrfBackground
        - Gaussian
        see also http://cta.irap.omp.eu/ctools/users/user_manual/models_spatial_bgd.html
        for further implementation.
        
        Parameters
        ----------
        - bkg_type (str): background type
        
        """
        
        if bkg_type == 'CTAIrfBackground':
            self.spatial  = {'type':'CTAIrfBackground'}
        
        elif bkg_type == 'Gaussian':
            self.spatial  = {'type':'Gaussian',
                             'param':{'Sigma':{'value':3.0*u.deg, 'free':False}}}
            
        else:
            raise ValueError('The baground type you are trying to set is not implemented.')
        
        
    #==================================================
    # Print the background
    #==================================================
    
    def print_bkg(self):
        """
        Show the background model.
        
        Parameters
        ----------
        
        """
        
        #----- First show the name
        print('--- name: '+self.name+' ---')

        #----- Show the spatial component
        print('--- Spatial model: '+self.spatial['type'])
        if 'param' in self.spatial.keys():
            for key in self.spatial['param'].keys():
                print('         '+key+': '+str(self.spatial['param'][key]))

        #----- Show the spectral component
        print('--- Spectral model: '+self.spectral['type'])        
        if 'param' in self.spectral.keys():
            for key in self.spectral['param'].keys():
                print('         '+key+': '+str(self.spectral['param'][key]))
