"""
This file contains the ClusterPipe class. It is dedicated to the construction of a 
ClusterPipe object, which defines how CTA observations of a cluster would proceed.
These simulations of observations are then performed with a quicklook analysis 
is available. The class provides an analysis pipeline to reduce the events.

"""


#==================================================
# Requested imports
#==================================================

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

import minot as model_cluster
from kesacco.Tools import compact_source as model_compsource
from kesacco.Tools import obs_setup      as setup_observations
from kesacco.clustpipe_common import Common
from kesacco.clustpipe_sim    import CTAsim
from kesacco.clustpipe_ana    import CTAana
from kesacco import clustpipe_title


#==================================================
# ClusterPipe class
#==================================================

class ClusterPipe(Common, CTAsim, CTAana):
    """ 
    ClusterPipe class. 
    This class defines a ClusterPipe object. This contain the sky model, as well
    as the observation setup. It can generate event files given the model and 
    provide an analysis pipeline to extract relevant information.
        
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
    - See Common, CTAsim, and CTAana sub-classes
    
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
        
        #---------- Analysis parameters
        # Likelihood method related
        self.method_stack  = True
        self.method_binned = True
        self.method_ana    = '3D' # ONOFF or 3D
        self.method_stat   = 'DEFAULT' # CSTAT, WSTAT, CHI2
        
        # Map related
        self.map_reso      = 0.02*u.deg
        self.map_coord     = SkyCoord(0.0, 0.0, frame="icrs", unit="deg")
        self.map_fov       = 10*u.deg
        self.map_UsePtgRef = True # Re-defines coordinates/FoV using pointings
        
        # Spectrum related
        self.spec_edisp    = False
        self.spec_ebinalg  = 'LOG'
        self.spec_enumbins = 10
        self.spec_emin     = 50*u.GeV
        self.spec_emax     = 100*u.TeV

        # Time related
        self.time_tmin     = None
        self.time_tmax     = None
        self.time_nbin     = 10

        # MCMC related
        self.mcmc_nwalkers = 10
        self.mcmc_nsteps   = 1000
        self.mcmc_burnin   = 100
        self.mcmc_conf     = 68.0
        self.mcmc_Nmc      = 100
        
        
