"""
This file contains parser to ctools for scripts related
to timing.
"""

#==================================================
# Imports
#==================================================

import ctools
import cscripts
import numpy as np
from kesacco.Tools import utilities


#==================================================
# Lightcurve
#==================================================

def lightcurve(inobs, inmodel, srcname, outfile,
               map_reso, map_coord, map_fov,
               xref, yref,
               caldb=None,
               irf=None,
               inexclusion=None,
               edisp=False,
               tbinalg='LIN',            
               tmin=None,
               tmax=None,
               mjdref=51544.5,
               tbins=20,
               tbinfile=None,
               method='3D',
               emin=1e-2, emax=1e+3, enumbins=20,
               rad=1.0,
               bkgregmin=2,
               use_model_bkg=True,
               maxoffset=4.0,
               etruemin=0.01,
               etruemax=0.01,
               etruebins=30,
               statistic='DEFAULT',
               calc_ts=True,
               calc_ulim=True,
               fix_srcs=True,
               fix_bkg=False,
               logfile=None,
               silent=False):    
    """
    Compute a lightcurve for a given source.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/cslightcrv.html

    Parameters
    ----------
    - inobs (str): Input event list or observation definition XML file.
    - inmodel (string): input model
    - srcname (str): name of the source to compute
    - outfile (str): output lightcurve file
    - map_reso (float): the resolution of the map (can be an
    astropy.unit object, or in deg)
    - map_coord (float): a skycoord object that give the center of the map
    - map_fov (float): the field of view of the map (can be an 
    astropy.unit object, or in deg)
    - xref/yref (float): the coordinates of the source (on/off) or the map. 
    - caldb (string): calibration database
    - irf (string): instrument response function
    - inexclusion (string): Optional FITS file containing a WCS map that 
    defines sky regions not to be used for background estimation (where 
    map value != 0). If the file contains multiple extensions the user may 
    specify which one to use. Otherwise, the extention EXCLUSION will be 
    used if available, or else the primary extension will be used.
    - edisp (bool): apply energy dispersion
    - tbinalg (string): <FILE|LIN|GTI> Algorithm for defining time bins.
    - tmin/tmax: Lightcurve start/stop time (UTC string, JD, MJD or MET in seconds).
    - mjdref (float): Reference Modified Julian Day (MJD) for Mission Elapsed Time (MET).
    - tbins (int): Number of time bins.
    - tbinfile (str): File defining the time binning.
    - method (string): <3D|ONOFF> Selects between 3D analysis (3D spatial/energy 
    likelihood) and ONOFF analysis (1D likelihood with background from Off regions).
    - emin,emax (float) min and max energy considered in TeV
    - enumbins (int): number of energy bins
    - rad (float): Radius of source region circle for On/Off analysis (deg)
    - bkgregmin (int): Minimum number of background regions that are required 
    for an observation in ONOFF analysis. If this number of background regions 
    is not available the observation is skipped.
    - use_model_bkg (bool): Specifies whether the background model should 
    be used for the computation of the alpha parameter and the predicted 
    background rate in the Off region that is stored in the BACKRESP column 
    of the Off spectrum when using the ONOFF method.
    - maxoffset(float): Maximum offset in degrees of source from camera 
    center to accept the observation for On/Off analysis.
    - etruemin (float): Minimum true energy to evaluate instrumental response 
    in On/Off analysis (TeV).
    - etruemax (float): Maximum true energy to evaluate instrumental response 
    in On/Off analysis (TeV).
    - etruebins (float): Number of bins per decade for true energy bins to 
    evaluate instrumental response in On/Off analysis.
    - statistic (str): Optimization statistic.
    - calc_ts (bool): Compute TS for each spectral point?
    - calc_ulim (bool): Compute upper limit for each spectral point?
    - fix_srcs (bool): Fix other sky model parameters?
    - fix_bkg (bool): Fix background model parameters?

    - silent (bool): print information or not

    Outputs
    --------
    - create lightcurve fits
    - return a lightcurve object

    """

    npix = utilities.npix_from_fov_def(map_fov, map_reso)

    lc = cscripts.cslightcrv()

    lc['inobs']    = inobs
    lc['inmodel']  = inmodel
    lc['srcname']  = srcname
    lc['outfile']  = outfile
    if caldb       is not None: lc['caldb']       = caldb     
    if irf         is not None: lc['irf']         = irf
    if inexclusion is not None: lc['inexclusion'] = inexclusion
    lc['edisp']    = edisp
    lc['tbinalg']  = tbinalg
    if tmin        is not None: lc['tmin']     = tmin
    if tmax        is not None: lc['tmax']     = tmax
    if tbinfile    is not None: lc['tbinfile'] = tbinfile
    lc['mjdref']        = mjdref
    lc['tbins']         = tbins
    lc['method']        = method
    lc['emin']          = emin 
    lc['emax']          = emax
    lc['enumbins']      = enumbins
    lc['nxpix']         = npix
    lc['nypix']         = npix
    lc['binsz']         = map_reso.to_value('deg')
    lc['coordsys']      = 'CEL'
    lc['proj']          = 'TAN'
    lc['xref']          = xref
    lc['yref']          = yref
    lc['srcshape']      = 'CIRCLE'
    lc['rad']           = rad
    lc['bkgmethod']     = 'REFLECTED'
    lc['bkgregmin']     = bkgregmin
    lc['use_model_bkg'] = use_model_bkg
    lc['maxoffset']     = maxoffset
    lc['etruemin']      = etruemin
    lc['etruemax']      = etruemax
    lc['etruebins']     = etruebins
    lc['statistic']     = statistic
    lc['calc_ts']       = calc_ts
    lc['calc_ulim']     = calc_ulim
    lc['fix_srcs']      = fix_srcs
    lc['fix_bkg']       = fix_bkg

    if logfile is not None: lc['logfile'] = logfile
    
    if logfile is not None: lc.logFileOpen()
    lc.execute()
    if logfile is not None: lc.logFileClose()
        
    if not silent:
        print(lc)
        print('')

    return lc
