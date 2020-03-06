"""
This file contains parser to ctools for scripts related
to imaging.
"""

#==================================================
# Imports
#==================================================

import ctools
import cscripts
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
           roiradius=0.1,
           inradius=1.0,
           outradius=2.0,
           iterations=3,
           threshold=3,
           silent=False):
    """
    Compute sky map.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/ctskymap.html

    Parameters
    ----------
    - inobs (str): input Event file or xml listing event file
    - outmap (str): full path to fits skymap
    - npix (int): number of pixels
    - reso(float): map resolution in deg
    - cra,cdec (float): map RA,Dec center in deg
    - emin,emax (float) min and max energy considered in TeV
    - caldb (str): calibration database
    - irf (str): instrument response function
    - bkgsubtract (str): 'NONE', 'IRF', or 'RING' method for 
    background subtraction
    - roiradius (float): for each pixel, radius around which on region is considered
    - inradius (float): inner radius for ring off region
    - outradius (float): outer radius for ring off region
    - iterations (int): number of iteration for flagging source regions
    - threshold (float): S/N threshold for flagging source region
    - silent (bool): print information or not

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

    if not silent:
        print(skymap)

    return skymap


#==================================================
# Source detect maps
#==================================================

def src_detect(inskymap, outmodel, outds9file,
               threshold=5.0,
               maxsrcs=20,
               avgrad=1.0,
               corr_rad=0.2,
               exclrad=0.2,
               silent=False):
    """
    Detect sources from a map
    See http://cta.irap.omp.eu/ctools/users/reference_manual/cssrcdetect.html

    Parameters
    ----------
    - inskymap (str): input skymap file
    - outmodel (str): output xml model filename
    - outds9file (str): output ds9 region filename
    - threshold (float): treshold (sigma gaussian) for detection
    - maxsrcs (int): maximum number of sources
    - avgrad (float): averaging radius for significance computation (degrees)
    - corr_rad (float): correlation kernel (deg)
    - exclrad (float): Radius around a detected source that is excluded 
    from further source detection (degrees).
    - silent (bool): print information or not

    Outputs
    --------
    - Write xml model file
    - Write a DS9 region file
    """
    
    srcdet = cscripts.cssrcdetect()    

    
    srcdet['inmap']      = inskymap
    srcdet['outmodel']   = outmodel
    srcdet['outds9file'] = outds9file
    srcdet['srcmodel']   = 'POINT'   # For the moment, only point source + power law
    srcdet['bkgmodel']   = 'NONE'    # NONE|IRF|AEFF|CUBE|RACC
    srcdet['threshold']  = threshold
    srcdet['maxsrcs']    = maxsrcs
    srcdet['avgrad']     = avgrad
    srcdet['corr_rad']   = corr_rad
    srcdet['corr_kern']  = 'DISK' #<NONE|DISK|GAUSSIAN>
    srcdet['exclrad']    = exclrad
    srcdet['fit_pos']    = True
    srcdet['fit_shape']  = True
        
    srcdet.execute()
    
    if not silent:
        print(srcdet)

    return srcdet


#==================================================
# TS map
#==================================================

def tsmap(inobs, inmodel, outmap, srcname,
          npix, reso, cra, cdec,
          expcube=None, 
          psfcube=None,
          bkgcube=None,
          edispcube=None,
          caldb=None,
          irf=None,
          edisp=False,
          statistic='DEFAULT',
          like_accuracy=0.005,
          max_iter=50,
          

        silent=False):
    """
    Compute TS map.
    http://cta.irap.omp.eu/ctools/users/reference_manual/cttsmap.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    tsmap = ctools.cttsmap()

    tsmap['inobs']         = inobs
    tsmap['inmodel']       = inmodel
    tsmap['srcname']       = srcname
    if expcube   is not None: tsmap['expcube']   = expcube
    if psfcube   is not None: tsmap['psfcube']   = psfcube
    if edispcube is not None: tsmap['edispcube'] = edispcube
    if bkgcube   is not None: tsmap['bkgcube']   = bkgcube
    if caldb     is not None: tsmap['caldb']     = caldb
    if irf       is not None: tsmap['irf']       = irf
    if edisp     is not None: tsmap['edisp']     = edisp
    tsmap['outmap']        = outmap
    tsmap['errors']        = False
    tsmap['statistic']     = statistic
    tsmap['like_accuracy'] = like_accuracy
    tsmap['max_iter']      = max_iter
    tsmap['usepnt']        = False
    tsmap['nxpix']         = npix
    tsmap['nypix']         = npix
    tsmap['binsz']         = reso
    tsmap['coordsys']      = 'CEL'
    tsmap['proj']          = 'TAN'
    tsmap['xref']          = cra
    tsmap['yref']          = cdec
    tsmap['binmin']        = -1
    tsmap['binmax']        = -1
    tsmap['logL0']         = -1.0
    
    tsmap.execute()
    
    if not silent:
        print(tsmap)
    
    return tsmap


#==================================================
# Residual maps
#==================================================

def resmap(inobs, inmodel, output_map,
           npix, reso, cra, cdec,
           emin=1e-2, emax=1e+3, enumbins=20, ebinalg='LOG',
           modcube=None, 
           expcube=None, 
           psfcube=None,
           bkgcube=None,
           edispcube=None,
           caldb=None,
           irf=None,
           edisp=False,
           algo='SIGNIFICANCE',
           silent=False):
    """
    Compute a residual map.
    
    Parameters
    ----------
    
    Outputs
    --------
    - create sky map fits
    - return a skymap object
    """

    resmap = cscripts.csresmap()
    
    resmap['inobs']     = inobs     
    resmap['inmodel']   = inmodel   
    if modcube   is not None: resmap['modcube']   = modcube   
    if expcube   is not None: resmap['expcube']   = expcube
    if psfcube   is not None: resmap['psfcube']   = psfcube
    if edispcube is not None: resmap['edispcube'] = edispcube
    if bkgcube   is not None: resmap['bkgcube']   = bkgcube
    if caldb     is not None: resmap['caldb']     = caldb     
    if irf       is not None: resmap['irf']       = irf       
    resmap['edisp']     = edisp     
    resmap['outmap']    = output_map 
    resmap['ebinalg']   = ebinalg   
    resmap['emin']      = emin      
    resmap['emax']      = emax      
    resmap['enumbins']  = enumbins  
    resmap['ebinfile']  = 'NONE'    
    resmap['coordsys']  = 'CEL'            
    resmap['proj']      = 'TAN'     
    resmap['xref']      = cra       
    resmap['yref']      = cdec      
    resmap['nxpix']     = npix      
    resmap['nypix']     = npix      
    resmap['binsz']     = reso      
    resmap['algorithm'] = algo      
    
    resmap.execute()

    if not silent:
        print(resmap)
    
    return resmap

