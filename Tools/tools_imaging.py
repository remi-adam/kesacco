"""
This file contains parser to ctools for scripts related
to imaging.
"""

#==================================================
# Imports
#==================================================

import ctools
import numpy as np

#==================================================
# Sky maps
#==================================================

def skymap(inobs, outmap,
           npix, reso, cra, cdec,
           emin=1e-2,
           emax=1e+3,
           caldb=None,
           irf=None,
           bkgsubtract='NONE',
           roiradius=2.0,
           inradius=3.0,
           outradius=4.0,
           iterations=3,
           threshold=3):
    """
    Compute sky map.

    Parameters
    ----------
    - See http://cta.irap.omp.eu/ctools/users/reference_manual/ctskymap.html

    Outputs
    --------
    - create sky map fits
    """
    
    skymap = ctools.ctskymap()    

    skymap['inobs']    = inobs
    if caldb is not None: skymap['caldb'] = caldb
    if irf   is not None: skymap['irf']   = irf
    skymap['inmap']    = 'NONE'
    skymap['outmap']   = outmap
    skymap['emin']     = emin
    skymap['emax']     = emax
    skymap['usepnt']   = False
    skymap['nxpix']    = npix
    skymap['nypix']    = npix
    skymap['binsz']    = reso
    skymap['coordsys'] = 'CEL'
    skymap['proj']     = 'TAN'
    skymap['xref']     = cra
    skymap['yref']     = cdec
    
    skymap['bkgsubtract'] = bkgsubtract
    skymap['roiradius']   = roiradius
    skymap['inradius']    = inradius
    skymap['outradius']   = outradius
    skymap['iterations']  = iterations
    skymap['threshold']   = threshold
    skymap['inexclusion'] = 'NONE'
    skymap['usefft']      = True
    
    skymap.execute()
    
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
