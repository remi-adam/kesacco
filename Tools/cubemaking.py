"""
This file contains parser to ctools for scripts related to 
binning the data.
"""

#==================================================
# Imports
#==================================================

import ctools
import numpy as np
import utilities

#==================================================
# Counts bin
#==================================================

def counts_cube(output_dir,
                map_reso, map_coord, map_fov,
                emin, emax, enumbins, ebinalg,
                stack=True):
    """
    Compute counts cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbin.html

    Parameters
    ----------

    Outputs
    --------
    """

    npix = utilities.npix_from_fov_def(map_fov, map_reso)
    
    binning = ctools.ctbin()
    
    binning['inobs']      = output_dir+'/AnaEventsSelected.xml'
    if stack:
        binning['outobs'] = output_dir+'/AnaCountscube.fits'
    else:
        binning['outobs'] = output_dir+'/AnaCountscube.xml'
    binning['stack']      = stack
    binning['prefix']     = output_dir+'/AnaCountscube'
    binning['ebinalg']    = ebinalg
    binning['emin']       = emin.to_value('TeV')
    binning['emax']       = emax.to_value('TeV')
    binning['enumbins']   = enumbins
    binning['ebinfile']   = 'NONE'
    binning['usepnt']     = False
    binning['nxpix']      = npix
    binning['nypix']      = npix
    binning['binsz']      = map_reso.to_value('deg')
    binning['coordsys']   = 'CEL'
    binning['proj']       = 'TAN'
    binning['xref']       = map_coord.icrs.ra.to_value('deg')
    binning['yref']       = map_coord.icrs.dec.to_value('deg')

    print(binning)

    binning.execute()
    
    return binning


#==================================================
# Exposure
#==================================================

def exp_cube(output_dir,
             map_reso, map_coord, map_fov,
             emin, emax, enumbins, ebinalg):
    """
    Compute a exposure cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctexpcube.html

    Parameters
    ----------

    Outputs
    --------
    """

    npix = utilities.npix_from_fov_def(map_fov, map_reso)
    
    expcube = ctools.ctexpcube()
    
    expcube['inobs']      = output_dir+'/AnaEventsSelected.xml'
    expcube['incube']     = output_dir+'/AnaCountscube.fits'
    #expcube['caldb']      = 
    #expcube['irf']        = 
    expcube['outcube']    = output_dir+'/AnaExpcube.fits'
    expcube['ebinalg']    = ebinalg
    expcube['emin']       = emin.to_value('TeV')
    expcube['emax']       = emax.to_value('TeV')
    expcube['enumbins']   = enumbins
    expcube['ebinfile']   = 'NONE'
    expcube['addbounds']  = False
    expcube['usepnt']     = False
    expcube['nxpix']      = npix
    expcube['nypix']      = npix
    expcube['binsz']      = map_reso.to_value('deg')
    expcube['coordsys']   = 'CEL'
    expcube['proj']       = 'TAN'
    expcube['xref']       = map_coord.icrs.ra.to_value('deg')
    expcube['yref']       = map_coord.icrs.dec.to_value('deg')

    print(expcube)
    expcube.execute()
    
    return expcube


#==================================================
# PSF cube
#==================================================

def psf_cube(output_dir,
             map_reso, map_coord, map_fov,
             emin, emax, enumbins, ebinalg,
             amax=0.3, anumbins=200):
    """
    Compute a PSF cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctpsfcube.html

    Parameters
    ----------

    Outputs
    --------
    """

    npix = utilities.npix_from_fov_def(map_fov, map_reso)
    
    psfcube = ctools.ctpsfcube()

    psfcube['inobs']      = output_dir+'/AnaEventsSelected.xml'
    psfcube['incube']     = output_dir+'/AnaCountscube.fits'
    #psfcube['caldb']      =
    #psfcube['irf']        =
    psfcube['outcube']    = output_dir+'/AnaPsfcube.fits'
    psfcube['ebinalg']    = ebinalg
    psfcube['emin']       = emin.to_value('TeV')
    psfcube['emax']       = emax.to_value('TeV')
    psfcube['enumbins']   = enumbins
    psfcube['ebinfile']   = 'NONE'
    psfcube['addbounds']  = False
    psfcube['usepnt']     = False
    psfcube['nxpix']      = npix
    psfcube['nypix']      = npix
    psfcube['binsz']      = map_reso.to_value('deg')
    psfcube['coordsys']   = 'CEL'
    psfcube['proj']       = 'TAN'
    psfcube['xref']       = map_coord.icrs.ra.to_value('deg')
    psfcube['yref']       = map_coord.icrs.dec.to_value('deg')
    psfcube['amax']       = amax
    psfcube['anumbins']   = anumbins

    print(psfcube)
    psfcube.execute()

    return psfcube


#==================================================
# Bkg cube
#==================================================

def bkg_cube(output_dir):
    """
    Compute a background cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbkgcube.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    bkgcube = ctools.ctbkgcube()

    bkgcube['inobs']    = output_dir+'/AnaEventsSelected.xml'
    bkgcube['incube']   = output_dir+'/AnaCountscube.fits'
    bkgcube['inmodel']  = output_dir+'/AnaModelInput.xml'
    #bkgcube['caldb']    =
    #bkgcube['irf']      =
    bkgcube['outcube']  = output_dir+'/AnaBkgcube.fits'
    bkgcube['outmodel'] = output_dir+'/AnaModelIntputStack.xml'

    print(bkgcube)
    bkgcube.execute()
    
    return bkgcube


#==================================================
# Edisp cube
#==================================================

def edisp_cube(output_dir,
               map_coord, map_fov,
               emin, emax, enumbins, ebinalg,
               binsz=1.0,migramax=2.0, migrabins=100):
    """
    Compute an energy dispersion cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctedispcube.html

    Parameters
    ----------

    Outputs
    --------
    """

    npix = utilities.npix_from_fov_def(map_fov.to_value('deg'), binsz)

    edc = ctools.ctedispcube()

    edc['inobs']      = output_dir+'/AnaEventsSelected.xml'
    edc['incube']     = 'NONE' #output_dir+'/AnaCountscube.fits'
    #edc['caldb']    = 
    #edc['irf']      =
    edc['outcube']    = output_dir+'/AnaEdispcube.fits'
    edc['ebinalg']    = ebinalg
    edc['emin']       = emin.to_value('TeV')
    edc['emax']       = emax.to_value('TeV')
    edc['enumbins']   = enumbins
    edc['ebinfile']   = 'NONE'
    edc['addbounds']  = False
    edc['usepnt']     = False
    edc['nxpix']      = npix
    edc['nypix']      = npix
    edc['binsz']      = binsz
    edc['coordsys']   = 'CEL'
    edc['proj']       = 'TAN'
    edc['xref']       = map_coord.icrs.ra.to_value('deg')
    edc['yref']       = map_coord.icrs.dec.to_value('deg')
    edc['migramax']   = migramax
    edc['migrabins']  = migrabins
    
    print(edc)
    edc.execute()
    
    return edc


#==================================================
# Mask cube
#==================================================

def mask_cube():
    """
    Compute a mask cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctcubemask.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    #maskcube = ctools.ctcubemask()
    #maskcube.execute()
    
    return maskcube


#==================================================
# Model cube
#==================================================

def model_cube():
    """
    Compute a model cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctmodel.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    model = ctools.ctmodel()

    model.execute()
    
    return model
