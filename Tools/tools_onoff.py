"""
This file contains parser to ctools for scripts related
to OnOff analysis, and other utilities
"""

#==================================================
# Imports
#==================================================

import ctools
import cscripts
import gammalib
from kesacco.Tools import utilities
import numpy as np
import astropy.units as u
from scipy.optimize import brentq
from astropy.io import fits
from minot.ClusterTools import map_tools

#==================================================
# Sky maps
#==================================================

def onoff_filegen(inobs, inmodel,
                  srcname,
                  outobs, outmodel,
                  prefix,
                  ebinalg, emin, emax, enumbins,
                  ra, dec, rad,
                  caldb=None, irf=None,
                  inexclusion=None,
                  bkgregmin=2, bkgregskip=1,
                  use_model_bkg=True,
                  maxoffset=4.0,
                  stack=True,
                  etruemin=0.01,
                  etruemax=300,
                  etruebins=30,
                  logfile=None,
                  silent=False):
    """
    Compute ON OFF regions.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/ctskymap.html

    Parameters
    ----------
    - 

    Outputs
    --------
    - onoff files are created
    - onoff cscript is output

    """

    #---------- Collect the bining info
    ebin_case, ebin_pow, ebin_file = utilities.get_binalg(ebinalg)
    
    #---------- Run script    
    onoff = cscripts.csphagen()
    
    # Input event list or observation definition XML file.
    onoff['inobs'] = inobs
    
    # Input model XML file (if NONE a point source at the centre of the source region is used).
    onoff['inmodel'] = inmodel

    # Name of the source in the source model XML file which should be used for ARF computation.
    onoff['srcname'] = srcname

    # Calibration database.
    if caldb is not None:
        onoff['caldb'] = caldb
    else:
        onoff['caldb'] = 'NONE'

    # Instrument response function.
    if irf is not None:
        onoff['irf'] = irf
    else:
        onoff['irf'] = 'NONE'

    # Optional FITS file containing a WCS map that defines sky regions not to be used for background estimation
    if inexclusion is not None:
        onoff['inexclusion'] = inexclusion
                                  
    # Output observation definition XML file.
    onoff['outobs'] = outobs
    
    # Output model XML file.
    onoff['outmodel'] = outmodel
    
    # Prefix of the file name for output PHA, ARF, RMF, XML, and DS9 region files.
    onoff['prefix'] = prefix
    
    # Algorithm for defining energy bins.
    onoff['ebinalg'] = ebin_case
    
    # Lower energy value for first energy bin (in TeV) if LIN or LOG energy binning algorithms are used.
    onoff['emin'] = emin
    
    # Upper energy value for last energy bin (in TeV) if LIN or LOG energy binning algorithms are used.
    onoff['emax'] = emax

    # Number of energy bins if LIN or LOG energy binning algorithms are used.
    onoff['enumbins'] = enumbins

    # Name of the file containing the energy binning definition
    onoff['ebinfile'] = ebin_file
    
    # Exponent of the power law for POW energy binning.
    onoff['ebingamma'] = ebin_pow
    
    # Shape of the source region. So far only CIRCLE exists which defines a circular region around location
    onoff['srcshape'] = 'CIRCLE'
    
    # Coordinate system (CEL - celestial, GAL - galactic).
    onoff['coordsys'] = 'CEL'
    
    # Right Ascension of source region centre (deg).
    onoff['ra'] = ra
    
    # Declination of source region centre (deg).
    onoff['dec'] = dec
    
    # Radius of source region circle (deg).
    onoff['rad'] = rad
    
    # Source region file (ds9 or FITS WCS map).
    onoff['srcregfile'] = 'NONE'
    
    # Method for background estimation
    onoff['bkgmethod'] = 'REFLECTED'
    
    # Background regions file (ds9 or FITS WCS map).
    onoff['bkgregfile'] = 'NONE'
    
    # Minimum number of background regions that are required for an observation.
    onoff['bkgregmin'] = bkgregmin
    
    # Number of background regions that should be skipped next to the On regions.
    onoff['bkgregskip'] = bkgregskip
    
    # Specifies whether the background model should be used for the computation of the alpha parameter
    onoff['use_model_bkg'] = use_model_bkg
    
    # Maximum offset in degrees of source from camera center to accept the observation.
    onoff['maxoffset'] = maxoffset
    
    # Specifies whether multiple observations should be stacked or whether run-wise files should be produced
    onoff['stack'] = stack
    
    # Minimum true energy (TeV).
    onoff['etruemin'] = etruemin
    
    # Maximum true energy (TeV).
    onoff['etruemax'] = etruemax

    # Number of bins per decade for true energy bins.
    onoff['etruebins'] = etruebins
        
    if logfile is not None: onoff['logfile'] = logfile

    if logfile is not None: onoff.logFileOpen()
    onoff.execute()
    if logfile is not None: onoff.logFileClose()

    if not silent:
        print(onoff)
        print('')

    return onoff


#==================================================
# Compute the source fraction in the on region
#==================================================

def on_source_fraction(cluster, radius, Emin, Emax):
    """
    Compute the source fraction in the on region, 
    assuming the current cluster model

    Parameters
    ----------
    - cluster (minot): minot cluster object
    - radius (quantity): radius of the on region homogeneous to deg
    - emin/emax: min and max energy for flux computation

    Outputs
    --------
    - fsrc (float): the fraction of source in the On region

    """

    Ftot = cluster.get_gamma_flux(Emin=Emin, Emax=Emax,
                                  Rmin=1*u.kpc, Rmax=cluster.R_truncation,
                                  type_integral='cylindrical')
    F_r = cluster.get_gamma_flux(Emin=Emin, Emax=Emax,
                                 Rmin=1*u.kpc, Rmax=cluster.R500/cluster.theta500*radius,
                                 type_integral='cylindrical')

    fsrc = (F_r / Ftot).to_value('')
    
    return fsrc


#==================================================
# Compute the angle so that X% of the flux is enclosed
#==================================================

def containment_on_source_fraction(cluster, fsrc, Emin, Emax):
    """
    Compute the radius so that fsrc fraction of the total flux is included
    in the on region.

    Parameters
    ----------
    - cluster (minot): minot cluster object
    - fsrc (float): fraction of flux source required
    - emin/emax: min and max energy for flux computation

    Outputs
    --------
    - theta (quantity): the radius so that the fraction of the 
    source inside is the one requested

    """

    # Compute the total flux
    Ftot = cluster.get_gamma_flux(Emin=Emin, Emax=Emax,
                                  Rmin=1*u.kpc, Rmax=cluster.R_truncation,
                                  type_integral='cylindrical')
    
    # defines the function where to search for roots
    def enclosed_flux_difference(rkpc):
        F_r = cluster.get_gamma_flux(Emin=Emin, Emax=Emax,
                                     Rmin=1*u.kpc, Rmax=rkpc*u.kpc,
                                     type_integral='cylindrical')
        func = (F_r - fsrc * Ftot).value
        return func

    # Search the root
    Rlim = brentq(enclosed_flux_difference, 10, cluster.R_truncation.to_value('kpc'))

    # Convert to angle
    theta = (Rlim/cluster.D_ang.to_value('kpc'))*u.rad
    
    return theta.to('deg')


#==================================================
# Build exclusion map for OFF region
#==================================================

def build_exclusion_map(compact_sources,
                        map_coord, map_reso, map_fov,
                        outfile,
                        rad=0.2*u.deg):
    """
    Compute a map with pixels that should be excluded from the off region
    set to 1 (others should be 0). For instance, point sources in the field.

    Parameters
    ----------
    - compact_source: structure containing the compact sources in the field
    - map_coord (coord): coordinates of the map center
    - map_reso (quantity): resolution of the map
    - map_fov (quantity): field of view size
    - outfile (str): output file name
    - rad (quantity): the radius to exclude around a source

    Outputs
    --------
    - the map file is saved
    - the map is output

    """

    #----- Get a standard header
    header = map_tools.define_std_header(map_coord.ra.to_value('deg'), map_coord.dec.to_value('deg'),
                                         map_fov.to_value('deg'), map_fov.to_value('deg'),
                                         map_reso.to_value('deg'))
    
    #----- Get the R.A.-Dec. map
    ra_map, dec_map = map_tools.get_radec_map(header)

    #----- Compute the initial exclusion map
    exclu = ra_map * 0

    #===== Loop over source
    Nsource = len(compact_sources.name)

    for isrc in range(Nsource):
        #----- Compute the radius map
        rad_isrc = map_tools.greatcircle(ra_map, dec_map,
                                         compact_sources.spatial[isrc]['param']['RA']['value'].to_value('deg'),
                                         compact_sources.spatial[isrc]['param']['DEC']['value'].to_value('deg'))
        
        #----- Apply the mask
        wsource = rad_isrc <= rad.to_value('deg')
        exclu[wsource] = 1
        
    #----- Save the map
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = exclu
    hdu.header.add_comment('OFF region exclusion map')
    hdu.header.add_comment('Flagged region are 1, 0 is ok')
    hdu.writeto(outfile, overwrite=True)
    
    return exclu


#==================================================
# Make the onoff source model
#==================================================

def make_onoff_source_model(inmodel, outmodel, compact_source, keepbkg=True):
    """
    Make the source model for the on/off file generation

    Parameters
    ----------
    - inmodel (str): full path to initial model to use
    - outmodel (str): full path to output model to save
    - compact_source (see class in kesacco): compact source structure
    - keepbkg (bool): keep or not the background in the file

    Outputs
    --------
    - output model saved

    """

    model = gammalib.GModels(inmodel)

    # Remove compact sources
    for iscr in range(len(compact_source.name)):
        model.remove(compact_source.name[iscr])

    # Remove background if needed
    if keepbkg == False:
        list2rm = []
        for imod in range(len(model)):
            cond1 = 'Background' in model[imod].name()
            cond2 = 'background' in model[imod].name()
            cond3 = 'BACKGROUND' in model[imod].name()
            if cond1 or cond2 or cond3:
                list2rm.append(model[imod].name())
                
        for imod in range(len(list2rm)):
            model.remove(list2rm[imod])

    # Save model
    model.save(outmodel)


#==================================================
# Rename the background model
#==================================================

def rename_bkg_onoff(modname):
    """
    Change the background name in the case of stacked background.

    Parameters
    ----------
    - modname (str): full path to model to change

    Outputs
    --------
    - output model saved

    """

    # Read the model
    model = gammalib.GModels(modname)

    # Loop over models to search for background
    ctn = 0
    for imod in range(len(model)):
        cond1 = 'Background' in model[imod].name()
        cond2 = 'background' in model[imod].name()
        cond3 = 'BACKGROUND' in model[imod].name()
        # Change the name when background is found
        if cond1 or cond2 or cond3:
            ctn += 1
            model[imod].name('BackgroundModel')

    # One and only one background model should be found
    if ctn != 1:
        print('WARNING: multiple background found in stacked observations after ONOFF file gen')

    # Save the model
    model.save(modname)
            
