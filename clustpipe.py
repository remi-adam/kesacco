"""
This file contains the ClusterPipe class. It is dedicated to the construction of a 
ClusterPipe object, which defines how CTA observations of a cluster would proceed.
These simulations of observations are then performed with a quicklook analysis 
is available. The class provides an analysis pipeline to reduce the events.

"""

#==================================================
# Requested imports
#==================================================

import os
import copy
import numpy as np
import pickle
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

from ClusterModel      import model          as model_cluster
from ClusterPipe.Tools import compact_source as model_compsource
from ClusterPipe.Tools import obs_setup      as setup_observations
from ClusterPipe.Tools import utilities

from ClusterPipe.clustpipe_admin import Admin
from ClusterPipe.clustpipe_sim   import CTAsim
from ClusterPipe.clustpipe_ana   import CTAana

from ClusterPipe import clustpipe_title


#==================================================
# ClusterPipe class
#==================================================

class ClusterPipe(Admin, CTAsim, CTAana):
    """ 
    ClusterPipe class. 
    This class defines a ClusterPipe object. This contain the sky model, as well
    as the observation setup. It can generate event files given the model and 
    provide an analysis pipeline to extract relevant information.
    
    To do list
    ----------
    
    Attributes
    ----------  
    - silent (bool): print information if False, or not otherwise.
    - output_dir (str): directory where to output data files and plots.
    - cluster (ClusterModel object): the cluster object which gather the
    physical properties of clusters
    - compact_source (CompactSource object): object from the class CompactSource
    which gather the properties of compact sources (i.e. not the cluster)
    in the region of interest
    - obs_setup (ObsSetup object): object from the ObsSetup class
    which gather the observation setup
    
    Methods
    ----------  
    
    """
    
    #==================================================
    # Initialize the CTAsim object
    #==================================================

    def __init__(self,
                 silent=False,
                 output_dir='./KESACCO',
                 cluster        = model_cluster.Cluster(silent=True),
                 compact_source = model_compsource.CompactSource(),
                 obs_setup      = setup_observations.ObsSetup()):
        """
        Initialize the ClusterPipe object.
        
        Parameters
        ----------
        - silent (bool): set to true in order not to print informations when running 
        - output_dir (str): where to save outputs
        - cluster: Cluster object can be passed here directly
        - compact_source: CompactSource object can be passed here directly
        - obs_setup: ObsSetup object can be passed here directly
        
        """
        
        #---------- Print the code header at launch
        if not silent:
            clustpipe_title.show()
        
        #---------- Admin
        self.silent     = silent
        self.output_dir = output_dir
        cluster.output_dir = output_dir
        
        #---------- Sky model
        self.cluster        = cluster
        self.compact_source = compact_source

        #---------- Observations (including background)
        self.obs_setup = obs_setup
        
        #---------- Ananlysis parameters
        # Map related
        self.map_reso     = 0.1*u.deg
        self.map_coord    = SkyCoord(0.0, 0.0, frame="icrs", unit="deg")
        self.map_fov      = 10*u.deg
        self.map_coordsys = 'CEL'
        self.map_proj     = 'TAN'
        
        # Likelihood method related
        self.method_stack  = True
        self.method_binned = True
        self.method_stat   = 'DEFAULT' # CSTAT, WSTAT, CHI2
        self.method_onoff  = False

        # Spectrum related
        self.ebinalg  = 'LOG'
        self.enumbins = 10
        self.emin     = 50*u.GeV
        self.emax     = 100*u.TeV
        
        
    #==================================================
    # Load the simulation configuration
    #==================================================
    
    def match_cluster_to_pointing(self, extra=1.1):
        """
        Match the cluster map and according to the pointing list.
        
        Parameters
        ----------
        - extra (float): factor to apply to the cluster extent to 
        have a bit of margin on the side of the map

        Outputs
        -------
        The cluster map properties are modified
        
        """

        # Compute the pointing barycenter and the FoV requested size 
        list_ptg_coord = SkyCoord(self.obs_setup.coord)
        list_ptg_rad   = self.obs_setup.rad
        center_ptg, fov_ptg = utilities.listcord2fov(list_ptg_coord, list_ptg_rad)

        # Account for the cluster
        fov = utilities.squeeze_fov(center_ptg, fov_ptg,
                                    self.cluster.coord, self.cluster.theta_truncation,
                                    extra=extra)

        # Set the cluster map to match the pointing
        self.cluster.map_coord = center_ptg
        self.cluster.map_fov   = fov
        
        
