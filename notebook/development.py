"""
This the equivalent of the notebook development.ipynb, but in a python script.
It provides an example to the Cluster simulation and analysis with CTA.
"""

##################################################################
# Imports
##################################################################

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
import numpy as np
import copy

from kesacco import clustpipe


##################################################################
# Parameters
##################################################################

#=================================================================
#======= Define the simulation
#=================================================================
output_dir = '/pbs/home/r/radam/Project/CTA/Phys/Outputs/KESACCO_example'

cpipe = clustpipe.ClusterPipe(silent=False, output_dir=output_dir)
print('')

#=================================================================
#======= Define the cluster object
#=================================================================

# Set the cluster basic properties
cpipe.cluster.name     = 'Perseus'
cpipe.cluster.redshift = 0.017284
cpipe.cluster.M500     = 6.2e14*u.solMass # Used for getting R500 and the pressure profile
cpipe.cluster.coord    = SkyCoord("3h19m47.2s +41d30m47s", frame='icrs')

# Truncate the cluster at 1.5 deg to allow small map field of view (for faster code) without loss of flux
cpipe.cluster.theta_truncation = 1.5*u.deg

# The target gas density [Churazov et al. 2003]
cpipe.cluster.density_gas_model = {'name':'doublebeta', 'beta1':1.2, 'r_c1':57*u.kpc, 'n_01':0.046*u.cm**-3,
                                   'beta2':0.71, 'r_c2':278*u.kpc, 'n_02':0.0036*u.cm**-3}

# The thermal profile (assuming Planck 2013 UPP)
cpipe.cluster.set_pressure_gas_gNFW_param('P13UPP')

# CR physics
cpipe.cluster.X_cre1_E = {'X':0.0, 'R_norm':cpipe.cluster.R500} # No primary CRe
cpipe.cluster.spectrum_crp_model = {'name':'PowerLaw', 'Index':2.2}
cpipe.cluster.set_density_crp_isobaric_scal_param(scal=0.5)     # Assumes CR follow n_CR \propto P_th^0.5
cpipe.cluster.X_crp_E = {'X':1.0, 'R_norm':cpipe.cluster.R500} # Assumes large amount of CR (~0.01 expected)

# Sampling
cpipe.cluster.map_reso = 0.01*u.deg      # Ideally should be few times smaller than the PSF
cpipe.cluster.Npt_per_decade_integ = 30

# Get information about the state of the cluster model
cpipe.cluster.print_param()
print('')


#=================================================================
#======= Define the compact sources
#=================================================================

# source 1
name='NGC1275'
spatial={'type':'PointSource',
         'param':{'RA': {'value':SkyCoord("3h19m48.16s +41d30m42s").ra.to('deg'),  'free':False},
                  'DEC':{'value':SkyCoord("3h19m48.16s +41d30m42s").dec.to('deg'), 'free':False}}}
spectral={'type':'PowerLaw',
          'param':{'Prefactor':{'value':2.1e-11/u.cm**2/u.TeV/u.s, 'free':True},
                   'Index':{'value':-3.6, 'free':True},
                   'PivotEnergy':{'value':0.2*u.TeV, 'free':False}}}

cpipe.compact_source.add_source(name, spatial, spectral) # Add the source to the model

# source 2
name='IC310'
spatial={'type':'PointSource',
         'param':{'RA': {'value':SkyCoord("3h16m42.98s +41d19m30s").ra.to('deg'),  'free':False},
                  'DEC':{'value':SkyCoord("3h16m42.98s +41d19m30s").dec.to('deg'), 'free':False}}}
spectral={'type':'PowerLaw',
          'param':{'Prefactor':{'value':4.3e-12/u.cm**2/u.TeV/u.s, 'free':True},
                   'Index':{'value':-1.95, 'free':True},
                   'PivotEnergy':{'value':1.0*u.TeV, 'free':False}}}

cpipe.compact_source.add_source(name, spatial, spectral)# Add the source to the model

# Show the status of the compact sources in the sky model
cpipe.compact_source.print_source()
print('')


#=================================================================
#======= Define the observation setup
#=================================================================

# One pointing offset +0 +1
cpipe.obs_setup.add_obs(obsid='001', name='Perseus_Ptg1', 
                        coord=SkyCoord(cpipe.cluster.coord.ra.value+0,
                                       cpipe.cluster.coord.dec.value+1, unit='deg'),
                        rad=5*u.deg,
                        emin=0.05*u.TeV, emax=100*u.TeV,
                        caldb='prod3b-v2', irf='North_z20_S_5h',
                        tmin='2020-01-01T00:00:00.0', tmax='2020-01-01T01:00:00.0', deadc=0.95)

# One pointing offset +sqrt(3)/2 -0.5
cpipe.obs_setup.add_obs(obsid='002', name='Perseus_Ptg2', 
                        coord=SkyCoord(cpipe.cluster.coord.ra.value+np.sqrt(3)/2,
                                       cpipe.cluster.coord.dec.value-0.5, unit='deg'),
                        rad=5*u.deg, 
                        emin=0.05*u.TeV, emax=100*u.TeV,
                        caldb='prod3b-v2', irf='North_z20_S_5h',
                        tmin='2020-01-02T00:00:00.0', tmax='2020-01-02T01:00:00.0', deadc=0.95)

# One pointing offset -sqrt(3)/2 -0.5
cpipe.obs_setup.add_obs(obsid='003', name='Perseus_Ptg3', 
                        coord=SkyCoord(cpipe.cluster.coord.ra.value-np.sqrt(3)/2,
                                       cpipe.cluster.coord.dec.value-0.5, unit='deg'),
                        rad=5*u.deg,
                        emin=0.05*u.TeV, emax=100*u.TeV,
                        caldb='prod3b-v2', irf='North_z20_S_5h',
                        tmin='2020-01-03T00:00:00.0', tmax='2020-01-03T01:00:00.0', deadc=0.95)

# Print info
cpipe.obs_setup.print_obs()
print('')


##################################################################
# Run the simulation
##################################################################

cpipe.run_sim_obs()
cpipe.run_sim_quicklook(ShowEvent=True,
                        ShowObsDef=True,
                        ShowSkyModel=True,
                        bkgsubtract='NONE',
                        smoothing_FWHM=0.2*u.deg)

print('')


##################################################################
# Run the analysis
##################################################################

#=================================================================
# Define the analysis setup
#=================================================================

#----- Method
cpipe.method_binned = True   # Do a binned analysis
cpipe.method_stack  = True   # Stack the event from different observations in a single analysis?
cpipe.method_ana    = '3D'   # 3D or ONOFF analysis

#----- Energy range/binning
cpipe.spec_enumbins = 10
cpipe.spec_emin     = 50*u.GeV
cpipe.spec_emax     = 10*u.TeV

#----- Imaging
# Force the use of the user defined map grid
cpipe.map_UsePtgRef     = False
# Define the map used for the binned analysis 
cpipe.map_reso          = 0.05*u.deg # Can be increaded if the code is too slow 
cpipe.map_fov           = 3*u.deg    # Can also be reduced (but should increase bkg-cluster degeneracy)
cpipe.map_coord         = copy.deepcopy(cpipe.cluster.coord)

#----- Define the map used for the template
cpipe.cluster.map_fov   = 2.1*cpipe.cluster.theta_truncation
cpipe.cluster.map_coord = copy.deepcopy(cpipe.cluster.coord)

#=================================================================
# Analysis
#=================================================================

cpipe.run_ana_dataprep()
cpipe.run_ana_likelihood()
cpipe.run_ana_imaging()
cpipe.run_ana_spectral()
cpipe.run_ana_expected_output()
cpipe.run_ana_plot()

cpipe.run_ana_mcmc_spectrum(reset_mcmc=True, run_mcmc=True, GaussLike=False)
cpipe.run_ana_mcmc_spectralimaging(reset_modelgrid=True,
                                   reset_mcmc=True, run_mcmc=True, GaussLike=False,
                                   spatial_range=[0.0,2.0], spatial_npt=11,
                                   spectral_range=[2.0,3.0], spectral_npt=11,
                                   bkg_marginalize=False, bkg_spectral_npt=13,
                                   bkg_spectral_range=[-0.05,0.05],ps_spectral_npt=19,
                                   ps_spectral_range=[-1.0,1.0],
                                   rm_tmp=False,FWHM=0.1*u.deg,theta=1.0*u.deg,coord=None,
                                   profile_reso=0.05*u.deg,includeIC=False)
