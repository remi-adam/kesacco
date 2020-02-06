"""
This file contains parser to ctools for scripts related
to maps.
"""

#==================================================
# Imports
#==================================================

import ctools
import numpy as np

#==================================================
# Sky maps
#==================================================

def skymap(output_map, evfile,
           caldb, irf,
           cra, cdec,
           npix=301,
           reso=0.05,
           emin_TeV=0.0,
           emax_TeV=np.inf,
           bkgsubtract=False):
    """
    Compute sky map.

    Parameters
    ----------
    - output_map (str): output map file
    - evfile (str): eventfile (full name)
    - caldb (str):calibration database
    - irf (str): input response function
    - cra, cdec (deg): RA and Dec centers
    - npix (int): the number of pixels along x and y (squared map)
    - reso (deg): the spatial bining resolution
    - emin_TeV/emax_TeV (TeV): energy range for the skymap
    - bkgsubtract (bool): subtract of not the background

    Outputs
    --------
    - create sky map fits
    - return a skymap object
    """
    
    #---------- Decide BKG subtraction
    if bkgsubtract == True: 
        bkgsub = 'IRF' # 'RING'
    else:
        bkgsub = 'NONE'
    
    #---------- Compute skymap
    skymap = ctools.ctskymap()    
    skymap['inobs']       = evfile
    skymap['caldb']       = caldb
    skymap['irf']         = irf
    skymap['outmap']      = output_map
    skymap['emin']        = emin_TeV
    skymap['emax']        = emax_TeV
    skymap['usepnt']      = False
    skymap['nxpix']       = npix
    skymap['nypix']       = npix
    skymap['binsz']       = reso
    skymap['proj']        = 'TAN'
    skymap['coordsys']    = 'CEL'
    skymap['xref']        = cra
    skymap['yref']        = cdec
    skymap['bkgsubtract'] = bkgsub
    
    skymap.run()
    skymap.save()
    
    return skymap


#==================================================
# Residual maps
#==================================================

def resmap(output_map, inobs, model, caldb, irf, cra, cdec,
           binned=False,
           modcube='NONE',
           expcube='NONE',
           psfcube='NONE', 
           edispcube='NONE',
           bkgcube='NONE',
           edisp=False,
           ebinalg='LOG',
           emin_TeV=0.0,
           emax_TeV=np.inf,
           enumbins=20,
           npix=301,
           reso=0.05,
           algo='SIGNIFICANCE'):
    """
    Compute a residual map.

    Parameters
    ----------
    - output_dir (str): output directory
    - inobs (str): eventfile (full name) or counts cube (for binning case)
    - caldb (str):calibration database
    - irf (str): input response function
    - cra, cdec (deg): RA and Dec centers
    - edisp (bool): apply energy dispersion
    - ebinalg (str): kind of energy bining (FILE|LIN|LOG)
    - emin_TeV/emax_TeV (TeV): energy range for the skymap
    - enumbins (int): the number of bins
    - npix (int): the number of pixels along x and y (squared map)
    - reso (deg): the spatial bining resolution
    - algo (str): algorithm to use (SUB, SUBDIV, SUBDIVSQRT, SIGNIFICANCE)

    Outputs
    --------
    - create sky map fits
    - return a skymap object
    """
    
    resmap = cscripts.csresmap()

    resmap['inobs']     = inobs
    resmap['inmodel']   = model

    if binned:
        resmap['modcube']   = modcube
        resmap['expcube']   = expcube
        resmap['psfcube']   = psfcube
        resmap['edispcube'] = edispcube
        resmap['bkgcube']   = bkgcube
    else:
        resmap['modcube']   = 'NONE'
        resmap['expcube']   = 'NONE'
        resmap['psfcube']   = 'NONE'
        resmap['edispcube'] = 'NONE'
        resmap['bkgcube']   = 'NONE'
        
    resmap['caldb']     = caldb
    resmap['irf']       = irf
    resmap['edisp']     = edisp
    resmap['outmap']    = output_map
    resmap['ebinalg']   = ebinalg
    resmap['emin']      = emin_TeV
    resmap['emax']      = emax_TeV
    resmap['enumbins']  = enumbins
    resmap['coordsys']  = 'CEL'
    resmap['proj']      = 'TAN'
    resmap['xref']      = cra
    resmap['yref']      = cdec
    resmap['nxpix']     = npix
    resmap['nypix']     = npix
    resmap['binsz']     = reso
    resmap['algorithm'] = algo
    resmap.run()
    resmap.save()

    return resmap


#==================================================
# TS map
#==================================================

def tsmap():
    """
    Compute TS map.

    Parameters
    ----------

    Outputs
    --------
    """
    
    
    return tsmap
