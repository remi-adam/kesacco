"""
This script is dedicated to lunch all the submodules that 
are necessary for the cluster simulation observations.
"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from random import randint
import astropy.units as u
from astropy.io import fits
import gammalib
import ctools

from ClusterSimCTA.Common import make_cluster_template
from ClusterSimCTA.Common import build_ctools_model


#==================================================
# Run the event simulation
#==================================================

def eventsim(output_dir, setup_obs, silent=False):
    """
    Run the event simulation with ctools

    Parameters
    ----------
    - output_dir (str): where to save outputs
    - setup_obs (object): define the observationnal setup of the run
    It should contain a single element in the list.
    - silent (bool): provide information or not
    
    Outputs
    --------
    - event file created
    """

    #----- Get the seed for reapeatable simu
    if setup_obs.seed[0] is None:
        seed = randint(1, 1e6)
    else:
        seed = setup_obs.seed[0]
    
    #----- Fill the simulation parameters
    simobs = ctools.ctobssim()
    
    simobs['inmodel']    = output_dir+'/Model.xml'
    simobs['caldb']      = setup_obs.caldb[0]
    simobs['irf']        = setup_obs.irf[0]
    simobs['edisp']      = setup_obs.edisp[0]
    simobs['outevents']  = output_dir+'/Events.fits'
    simobs['prefix']     = ''
    simobs['startindex'] = 1
    simobs['seed']       = seed
    simobs['ra']         = setup_obs.coord[0].icrs.ra.to_value('deg')
    simobs['dec']        = setup_obs.coord[0].icrs.dec.to_value('deg')
    simobs['rad']        = setup_obs.rad[0].to_value('deg')
    simobs['tmin']       = setup_obs.tmin[0]
    simobs['tmax']       = setup_obs.tmax[0]
    simobs['emin']       = setup_obs.emin[0].to_value('TeV')
    simobs['emax']       = setup_obs.emax[0].to_value('TeV')
    simobs['deadc']      = setup_obs.deadc[0]
    simobs['maxrate']    = 1e6

    #----- Run the simulation
    if not silent:
        print('   ------- Simulation log -------')
        print(simobs)
        print('')
    
    simobs.execute()

    
#==================================================
# Run the event simulation
#==================================================

def edit_header(filename, obj_name, obj_ra, obj_dec, obsid):
    """
    Edit the header to make the keywords compliant with the
    simulation that is done.

    Parameters
    ----------
    - file
    - obj_name,ra,dec, obsid: keywords to update
    
    Outputs
    --------
    - event file modified
    """

    with fits.open(filename, mode='update') as filehandle:
        filehandle[1].header['OBJECT']  = obj_name
        filehandle[1].header['RA_OBJ']  = obj_ra 
        filehandle[1].header['DEC_OBJ'] = obj_dec
        filehandle[1].header['OBS_ID']  = obsid

        
#==================================================
# Run the simulation
#==================================================

def run(output_dir,
        cluster_mapfile,
        cluster_specfile,
        cluster,
        compact_source,
        setup_obs,
        silent=False):
    """
    Main script of the cluster simulation pipeline, which run sub-modules
    for each individual run.

    Parameters
    ----------
    - output_dir (str): where to save outputs
    - compact_source (object): defines the point source properties
    - setup_obs (object): define the observationnal setup of the run
    - silent (bool): provide information or not
    
    Outputs
    --------
    - event file (+ associated products)
    """

    #========== Information
    if not silent:
        print('')
        print('---------- SIMULATON STARTING ----------')
        setup_obs.print_obs()
        print('')
        
    #========== Build the model
    model_tot = gammalib.GModels()
    
    #----- Build the cluster model
    build_ctools_model.cluster(model_tot,
                               cluster_mapfile,
                               cluster_specfile,
                               ClusterName=cluster.name)
    
    #----- Build point source model
    build_ctools_model.compact_sources(model_tot, compact_source)
    
    #----- Build background model
    build_ctools_model.background(model_tot, setup_obs.bkg[0])
    
    #----- Total model
    model_tot.save(output_dir+'/Model.xml')

    if not silent:
        print(model_tot)
        print('')
    
    #========== Make an observation definition file
    setup_obs.write_pnt(output_dir+'/Pnt.def')
    setup_obs.run_csobsdef(output_dir+'/Pnt.def', output_dir+'/ObsDef.xml')
    
    #========== Run the simulation
    eventsim(output_dir, setup_obs, silent=silent)

    #========== Edit the header
    edit_header(output_dir+'/Events.fits',
                setup_obs.name[0],
                cluster.coord.icrs.ra.to_value('deg'),
                cluster.coord.icrs.dec.to_value('deg'),
                setup_obs.obsid[0])
    
