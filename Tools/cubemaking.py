"""
This file contains parser to ctools for scripts related to 
binning the data.
"""

#==================================================
# Imports
#==================================================

import ctools
import numpy as np
from kesacco.Tools import utilities

#==================================================
# Counts bin
#==================================================

def counts_cube(output_dir,
                map_reso, map_coord, map_fov,
                emin, emax, enumbins, ebinalg,
                stack=True,
                logfile=None,
                silent=False):
    """
    Compute counts cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbin.html

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - map_reso (float): the resolution of the map (can be an
    astropy.unit object, or in deg)
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - emin/emax (float): min and max energy in TeV
    - enumbins (int): the number of energy bins
    - ebinalg (str): the energy binning algorithm
    - stack (bool): do we use stacking of individual event files or not
    - silent (bool): use this keyword to print information

    Outputs
    --------
    - Ana_Countscube.fits: the fits file cubed data, in case stack is requested
    - Ana_Countscube.xml: the xml file cubed data, in case stack is not requested
    - Ana_Countscubecta_{obsIDs}.fits: the fits files of individual event files
    cubed data, in case stack is not requested

    """
    
    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Pixels
    npix = utilities.npix_from_fov_def(map_fov, map_reso)

    #---------- Run script
    ctscube = ctools.ctbin()
    
    ctscube['inobs']      = output_dir+'/Ana_EventsSelected.xml'
    if stack:
        ctscube['outobs'] = output_dir+'/Ana_Countscube.fits'
    else:
        ctscube['outobs'] = output_dir+'/Ana_Countscube.xml'
    ctscube['stack']      = stack
    ctscube['prefix']     = output_dir+'/Ana_Countscube'
    ctscube['ebinalg']    = ebin_case
    ctscube['emin']       = emin.to_value('TeV')
    ctscube['emax']       = emax.to_value('TeV')
    ctscube['enumbins']   = enumbins
    ctscube['ebinfile']   = ebin_file
    ctscube['ebingamma']  = ebin_pow
    ctscube['usepnt']     = False
    ctscube['nxpix']      = npix
    ctscube['nypix']      = npix
    ctscube['binsz']      = map_reso.to_value('deg')
    ctscube['coordsys']   = 'CEL'
    ctscube['proj']       = 'TAN'
    ctscube['xref']       = map_coord.icrs.ra.to_value('deg')
    ctscube['yref']       = map_coord.icrs.dec.to_value('deg')
    if logfile is not None: ctscube['logfile'] = logfile

    if logfile is not None: ctscube.logFileOpen()
    ctscube.execute()
    if logfile is not None: ctscube.logFileClose()

    if not silent:
        print(ctscube)
        print('')

    return ctscube


#==================================================
# Exposure
#==================================================

def exp_cube(output_dir,
             map_reso, map_coord, map_fov,
             emin, emax, enumbins, ebinalg,
             logfile=None,
             silent=False):
    """
    Compute a exposure cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctexpcube.html
    Note that the number of point in energy corresponds to the bin poles.

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - map_reso (float): the resolution of the map (can be an
    astropy.unit object, or in deg)
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - emin/emax (float): min and max energy in TeV
    - enumbins (int): the number of energy bins
    - ebinalg (str): the energy binning algorithm
    - silent (bool): use this keyword to print information

    Outputs
    --------
    - Ana_Expcube.fits: exposure fits as a fits file
    
    """

    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Pixels
    npix = utilities.npix_from_fov_def(map_fov, map_reso)

    #---------- Run script
    expcube = ctools.ctexpcube()
    
    expcube['inobs']      = output_dir+'/Ana_EventsSelected.xml'
    expcube['incube']     = output_dir+'/Ana_Countscube.fits'
    #expcube['caldb']      = 
    #expcube['irf']        = 
    expcube['outcube']    = output_dir+'/Ana_Expcube.fits'
    expcube['ebinalg']    = ebin_case
    expcube['emin']       = emin.to_value('TeV')
    expcube['emax']       = emax.to_value('TeV')
    expcube['enumbins']   = enumbins
    expcube['ebinfile']   = ebin_file
    expcube['ebingamma']  = ebin_pow
    expcube['addbounds']  = False
    expcube['usepnt']     = False
    expcube['nxpix']      = npix
    expcube['nypix']      = npix
    expcube['binsz']      = map_reso.to_value('deg')
    expcube['coordsys']   = 'CEL'
    expcube['proj']       = 'TAN'
    expcube['xref']       = map_coord.icrs.ra.to_value('deg')
    expcube['yref']       = map_coord.icrs.dec.to_value('deg')
    if logfile is not None: expcube['logfile'] = logfile

    if logfile is not None: expcube.logFileOpen()
    expcube.execute()
    if logfile is not None: expcube.logFileClose()
    
    if not silent:
        print(expcube)
        print('')
        
    return expcube


#==================================================
# PSF cube
#==================================================

def psf_cube(output_dir,
             map_reso, map_coord, map_fov,
             emin, emax, enumbins, ebinalg,
             amax=0.3, anumbins=200,
             logfile=None,
             silent=False,
             useincube=True):
    """
    Compute a PSF cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctpsfcube.html
    Note that the number of point in energy corresponds to the bin poles.

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - map_reso (float): the resolution of the map (can be an
    astropy.unit object, or in deg)
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - emin/emax (float): min and max energy in TeV
    - enumbins (int): the number of energy bins
    - ebinalg (str): the energy binning algorithm
    - amax (float): Upper bound of angular separation between true and 
    measued photon direction (in degrees).
    - anumbins (int): Number of angular separation bins.
    - silent (bool): use this keyword to print information

    Outputs
    --------
    - Ana_Psfcube.fits: the fits PSF cube image
    """

    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Pixels
    npix = utilities.npix_from_fov_def(map_fov, map_reso)

    #---------- Run script    
    psfcube = ctools.ctpsfcube()

    psfcube['inobs']      = output_dir+'/Ana_EventsSelected.xml'
    if useincube:
        psfcube['incube'] = output_dir+'/Ana_Countscube.fits'
    else:
        psfcube['incube'] = 'NONE'
    #psfcube['caldb']      =
    #psfcube['irf']        =
    psfcube['outcube']    = output_dir+'/Ana_Psfcube.fits'
    psfcube['ebinalg']    = ebin_case
    psfcube['emin']       = emin.to_value('TeV')
    psfcube['emax']       = emax.to_value('TeV')
    psfcube['enumbins']   = enumbins
    psfcube['ebinfile']   = ebin_file
    psfcube['ebingamma']  = ebin_pow
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
    if logfile is not None: psfcube['logfile'] = logfile

    if logfile is not None: psfcube.logFileOpen()
    psfcube.execute()
    if logfile is not None: psfcube.logFileClose()

    if not silent:
        print(psfcube)
        print('')

    return psfcube


#==================================================
# Bkg cube
#==================================================

def bkg_cube(output_dir,
             logfile=None,
             silent=False):
    """
    Compute a background cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbkgcube.html

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - silent (bool): use this keyword to print information

    Outputs
    --------
    - Ana_Bkgcube.fits: the fits bkg cube image
    - Ana_Model_Intput_Stack.xml: the xml input model model after
    including the stacked background
    """
    
    bkgcube = ctools.ctbkgcube()

    bkgcube['inobs']    = output_dir+'/Ana_EventsSelected.xml'
    bkgcube['incube']   = output_dir+'/Ana_Countscube.fits'
    bkgcube['inmodel']  = output_dir+'/Ana_Model_Input_Unstack.xml'
    #bkgcube['caldb']    =
    #bkgcube['irf']      =
    bkgcube['outcube']  = output_dir+'/Ana_Bkgcube.fits'
    bkgcube['outmodel'] = output_dir+'/Ana_Model_Input_Stack.xml'
    if logfile is not None: bkgcube['logfile'] = logfile

    if logfile is not None: bkgcube.logFileOpen()
    bkgcube.execute()
    if logfile is not None: bkgcube.logFileClose()
    
    if not silent:
        print(bkgcube)
        print('')
        
    return bkgcube


#==================================================
# Edisp cube
#==================================================

def edisp_cube(output_dir,
               map_coord, map_fov,
               emin, emax, enumbins, ebinalg,
               binsz=1.0, migramax=2.0, migrabins=100,
               logfile=None,
               silent=False):
    """
    Compute an energy dispersion cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctedispcube.html
    Note that the number of point in energy corresponds to the bin poles.

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - emin/emax (float): min and max energy in TeV
    - enumbins (int): the number of energy bins
    - ebinalg (str): the energy binning algorithm
    - binsz (float): map resolution in deg. Small variation of edisp, so 
    no need to set it to small values
    - migramax (float): Upper bound of ratio between reconstructed and 
    true photon energy.
    - migrabins (int): Number of migration bins.
    - silent (bool): use this keyword to print information

    Outputs
    --------
    - Ana_Edispcube.fits: the energy dispersion fits file
    """

    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Pixels
    npix = utilities.npix_from_fov_def(map_fov.to_value('deg'), binsz)

    #---------- Run script
    edcube = ctools.ctedispcube()

    edcube['inobs']      = output_dir+'/Ana_EventsSelected.xml'
    edcube['incube']     = 'NONE' #output_dir+'/Ana_Countscube.fits'
    #edcube['caldb']    = 
    #edcube['irf']      =
    edcube['outcube']    = output_dir+'/Ana_Edispcube.fits'
    edcube['ebinalg']    = ebin_case
    edcube['emin']       = emin.to_value('TeV')
    edcube['emax']       = emax.to_value('TeV')
    edcube['enumbins']   = enumbins
    edcube['ebinfile']   = ebin_file
    edcube['ebingamma']  = ebin_pow
    edcube['addbounds']  = False
    edcube['usepnt']     = False
    edcube['nxpix']      = npix
    edcube['nypix']      = npix
    edcube['binsz']      = binsz
    edcube['coordsys']   = 'CEL'
    edcube['proj']       = 'TAN'
    edcube['xref']       = map_coord.icrs.ra.to_value('deg')
    edcube['yref']       = map_coord.icrs.dec.to_value('deg')
    edcube['migramax']   = migramax
    edcube['migrabins']  = migrabins
    if logfile is not None: edcube['logfile'] = logfile

    if logfile is not None: edcube.logFileOpen()
    edcube.execute()
    if logfile is not None: edcube.logFileClose()
    
    if not silent:
        print(edcube)
        print('')
    
    return edcube


#==================================================
# Model cube
#==================================================

def model_cube(output_dir,
               map_reso, map_coord, map_fov,
               emin, emax, enumbins, ebinalg,
               edisp=False,
               stack=True,
               inmodel_usr=None,
               outmap_usr=None,
               logfile=None,
               silent=False):
    """
    Compute a model cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctmodel.html

    Parameters
    ----------
    - output_dir (str): directory where to get input files and 
    save outputs
    - map_reso (float): the resolution of the map (can be an
    astropy.unit object, or in deg)
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - emin/emax (float): min and max energy in TeV
    - enumbins (int): the number of energy bins
    - ebinalg (str): the energy binning algorithm
    - stack (bool): do we use stacking of individual event files or not
    - inmodel_usr (str): use this keyword to pass non default inmodel
    - outmap_usr (str): use this keyword to pass non default outmap
    - silent (bool): use this keyword to print information

    Outputs
    --------
    """
    
    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Pixels
    npix = utilities.npix_from_fov_def(map_fov, map_reso)

    #---------- Run script        
    model = ctools.ctmodel()
    
    if stack:
        model['inobs'] = output_dir+'/Ana_Countscube.fits'
    else:
        model['inobs'] = output_dir+'/Ana_ObsDef.xml'
    if inmodel_usr is None:
        model['inmodel']   = output_dir+'/Ana_Model_Output.xml'
    else:
        model['inmodel']   = inmodel_usr
    model['incube']    = 'NONE'
    model['expcube']   = 'NONE'
    model['psfcube']   = 'NONE'
    model['edispcube'] = 'NONE'
    model['bkgcube']   = 'NONE'
    if stack:
        model['incube']    = output_dir+'/Ana_Countscube.fits'
        model['expcube']   = output_dir+'/Ana_Expcube.fits'
        model['psfcube']   = output_dir+'/Ana_Psfcube.fits'
        if edisp:
            model['edispcube'] = output_dir+'/Ana_Edispcube.fits'
        model['bkgcube']   = output_dir+'/Ana_Bkgcube.fits'
        
    model['caldb']     = 'NONE'
    model['irf']       = 'NONE'
    model['edisp']     = edisp
    if outmap_usr is None:
        model['outcube']   = output_dir+'/Ana_Model_Cube.fits'
    else:
        model['outcube']   = outmap_usr
    model['ra']        = 'NONE'
    model['dec']       = 'NONE'
    model['rad']       = 'NONE'
    model['tmin']      = 'NONE'
    model['tmax']      = 'NONE'
    model['deadc']     = 'NONE'
    model['ebinalg']   = ebin_case
    model['emin']      = emin.to_value('TeV')
    model['emax']      = emax.to_value('TeV')
    model['enumbins']  = enumbins
    model['ebinfile']  = ebin_file
    model['ebingamma']  = ebin_pow
    model['usepnt']    = False
    model['nxpix']     = npix
    model['nypix']     = npix
    model['binsz']     = map_reso.to_value('deg')
    model['coordsys']  = 'CEL'
    model['proj']      = 'TAN'
    model['xref']      = map_coord.icrs.ra.to_value('deg')
    model['yref']      = map_coord.icrs.dec.to_value('deg')
    if logfile is not None: model['logfile'] = logfile

    if logfile is not None: model.logFileOpen()
    model.execute()
    if logfile is not None: model.logFileClose()
    
    if not silent:
        print(model)
        print('')
        
    return model


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
    
    maskcube = ctools.ctcubemask()
    maskcube.execute()
    
    return maskcube

