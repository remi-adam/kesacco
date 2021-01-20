"""
This file contains the ClusterPipe class. It is dedicated to the construction of a 
ClusterPipe object, which defines how CTA observations of a cluster would proceed.
These simulations of observations are then performed and a quicklook analysis 
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
    - See Common (clustpipe.common.py), CTAsim (clustpipe_sim.py), 
    and CTAana (clustpipe_ana.py) sub-classes.
    
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
        
        #========== Print the code header at launch
        if not silent:
            clustpipe_title.show()
        
        #========== Admin
        # Print information or not
        self.silent        = silent
        # The working output directory
        self.output_dir    = output_dir
        cluster.output_dir = output_dir
        
        #========== Sky model
        # The cluster object as a minot object
        self.cluster        = cluster
        # The background sky source model
        self.compact_source = compact_source

        #========== Observations (including background)
        self.obs_setup = obs_setup
        
        #========== Analysis parameters
        #----- Likelihood method related
        # To stack the different run into a single data file or fit all together
        self.method_stack  = True
        # Bined or unbinned analysis
        self.method_binned = True
        # ONOFF or 3D
        self.method_ana    = '3D'
        # CSTAT, WSTAT, CHI2: the statistics to use
        self.method_stat   = 'DEFAULT' 
        
        #----- Map related
        # The pixel size of the grid
        self.map_reso      = 0.02*u.deg
        # The map center coordinates
        self.map_coord     = SkyCoord(0.0, 0.0, frame="icrs", unit="deg")
        # The size of the map
        self.map_fov       = 10*u.deg
        # Re-defines coordinates/FoV using obs pointings
        self.map_UsePtgRef = True                                        
        
        #----- Spectrum related
        # Apply energy dispersion
        self.spec_edisp    = False
        # Energy binning algorithm: LOG or LIN
        self.spec_ebinalg  = 'LOG'
        # Number of bins
        self.spec_enumbins = 10
        # Minimum energy
        self.spec_emin     = 50*u.GeV
        # Maximum energy
        self.spec_emax     = 100*u.TeV

        #----- Time related
        # Reference MJD
        self.time_mjdref = 51544.5
        # Minimum time
        self.time_tmin = None
        # Maximum time
        self.time_tmax = None
        # Number of time bins
        self.time_nbin = 10

        #----- MCMC related
        # Number of MCMC walkers
        self.mcmc_nwalkers = 10
        # Number of MCMC steps
        self.mcmc_nsteps   = 1000
        # Number of sample in the burnin
        self.mcmc_burnin   = 100
        # MCMC confidence limit for extracted distributions
        self.mcmc_conf     = 68.0
        # Number of points for Monte Carlo resampling
        self.mcmc_Nmc      = 100
        
