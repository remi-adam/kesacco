"""
This file contains parser to ctools for scripts related
to spectra.
"""

import gammalib
import ctools
import cscripts


#==================================================
# Spectrum
#==================================================

def spectrum(inobs, inmodel, srcname, outfile,
             emin=1e-2, emax=1e+3, enumbins=20, ebinalg='LOG',
             expcube=None, 
             psfcube=None,
             bkgcube=None,
             edispcube=None,
             caldb=None,
             irf=None,
             edisp=False,
             method='SLICE',
             statistic='DEFAULT',
             calc_ts=True,
             calc_ulim=True,
             fix_srcs=True,
             fix_bkg=False,
             dll_sigstep=1,
             dll_sigmax=7,
             dll_freenodes=False,
             logfile=None,
             silent=False):    
    """
    Compute a spectrum for a given source.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/csspec.html

    Parameters
    ----------
    - inobs (string): input observation file
    - inmodel (string): input model
    - srcname (str): name of the source to compute
    - outfile (str): output spectrum file
    - emin,emax (float) min and max energy considered in TeV
    - enumbins (int): number of energy bins
    - ebinalg (string): energy bining algorithm
    - modcube (string): model map cube
    - expcube (string): exposure map cube
    - psfcube (string): psfcube 
    - bkgcube (string): background cube
    - edispcube (string): energy dispersion cube
    - caldb (string): calibration database
    - irf (string): instrument response function
    - edisp (bool): apply energy dispersion
    - method (str): Spectrum generation method
    - statistic (str): Optimization statistic.
    - calc_ts (bool): Compute TS for each spectral point?
    - calc_ulim (bool): Compute upper limit for each spectral point?
    - fix_srcs (bool): Fix other sky model parameters?
    - fix_bkg (bool): Fix background model parameters?
    - dll_sigstep (float): sigma steps for the likelihood scan
    - dll_sigmax (float): sigma max for the likelihood scan
    - dll_freenodes (bool): free nodes for method = NODES 
    - silent (bool): print information or not

    Outputs
    --------
    - create spectrum fits
    - return a spectrum object

    """

    spec = cscripts.csspec()

    spec['inobs']     = inobs 
    spec['inmodel']   = inmodel
    spec['srcname']   = srcname
    if expcube   is not None: spec['expcube']   = expcube
    if psfcube   is not None: spec['psfcube']   = psfcube
    if edispcube is not None: spec['edispcube'] = edispcube
    if bkgcube   is not None: spec['bkgcube']   = bkgcube
    if caldb     is not None: spec['caldb']     = caldb     
    if irf       is not None: spec['irf']       = irf       
    spec['edisp']     = edisp
    spec['outfile']   = outfile
    spec['method']    = method
    spec['ebinalg']   = ebinalg
    spec['emin']      = emin 
    spec['emax']      = emax
    spec['enumbins']  = enumbins
    spec['ebinfile']  = 'NONE'
    spec['statistic'] = statistic
    spec['calc_ts']   = calc_ts
    spec['calc_ulim'] = calc_ulim
    spec['fix_srcs']  = fix_srcs
    spec['fix_bkg']   = fix_bkg
    spec['dll_sigstep'] = dll_sigstep
    spec['dll_sigmax']  = dll_sigmax
    spec['dll_freenodes'] = dll_freenodes

    if logfile is not None: spec['logfile'] = logfile

    if logfile is not None: spec.logFileOpen()
    spec.execute()
    if logfile is not None: spec.logFileClose()
        
    if not silent:
        print(spec)
        print('')

    return spec


#==================================================
# Butterfly
#==================================================

def butterfly(inobs, inmodel, srcname, outfile,
              emin=5e-3, emax=5e+3, enumbins=100, ebinalg='LOG',
              expcube=None, 
              psfcube=None,
              bkgcube=None,
              edispcube=None,
              caldb=None,
              irf=None,
              edisp=False,
              fit=False,
              method='GAUSSIAN',
              confidence=0.68,
              statistic='DEFAULT',
              like_accuracy=0.005,
              max_iter=50,
              matrix='NONE',
              logfile=None,
              silent=False):
    """
    Computes butterfly diagram for a given spectral model.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/ctbutterfly.html

    Parameters
    ----------
    - inobs (string): input observation file
    - inmodel (string): input model
    - srcname (str): name of the source to compute
    - outfile (str): output spectrum file
    - emin,emax (float) min and max energy considered in TeV
    - enumbins (int): number of energy bins
    - ebinalg (string): energy bining algorithm
    - modcube (string): model map cube
    - expcube (string): exposure map cube
    - psfcube (string): psfcube 
    - bkgcube (string): background cube
    - edispcube (string): energy dispersion cube
    - caldb (string): calibration database
    - irf (string): instrument response function
    - edisp (bool): apply energy dispersion
    - fit (bool): Performs maximum likelihood fitting of input 
    model ignoring any provided covariance matrix.
    - method (string): method for butterfly contours
    - confidence (float): confidence limit
    - statistic (str): Optimization statistic.
    - like_accuracy (float): Absolute accuracy of maximum likelihood value
    - max_iter (int): Maximum number of fit iterations.
    - matrix (string): fir covariance matrix
    - silent (bool): print information or not

    Outputs
    --------
    - create butterfly ascii file

    """
    
    but = ctools.ctbutterfly()

    but['inobs']   = inobs
    but['inmodel'] = inmodel
    but['srcname'] = srcname
    if expcube   is not None: but['expcube']   = expcube
    if psfcube   is not None: but['psfcube']   = psfcube
    if edispcube is not None: but['edispcube'] = edispcube
    if bkgcube   is not None: but['bkgcube']   = bkgcube
    if caldb     is not None: but['caldb']     = caldb     
    if irf       is not None: but['irf']       = irf       
    but['edisp']         = edisp
    but['outfile']       = outfile
    but['fit']           = fit
    but['method']        = method
    but['confidence']    = confidence
    but['statistic']     = statistic
    but['like_accuracy'] = like_accuracy
    but['max_iter']      = max_iter
    but['matrix']        = matrix
    but['ebinalg']       = ebinalg
    but['emin']          = emin 
    but['emax']          = emax
    but['enumbins']      = enumbins
    but['ebinfile']      = 'NONE'
    if logfile is not None: but['logfile'] = logfile

    if logfile is not None: but.logFileOpen()
    but.execute()
    if logfile is not None: but.logFileClose()
        
    if not silent:
        print(but)
        print('')

    return but


#==================================================
# Spectrum residual
#==================================================

def residual(inobs, inmodel, outfile,
             npix, reso, cra, cdec,
             res_ra=None, res_dec=None, res_rad=1.0,
             emin=1e-2, emax=1e+3, enumbins=20, ebinalg='LOG',
             modcube=None, 
             expcube=None, 
             psfcube=None,
             bkgcube=None,
             edispcube=None,
             caldb=None,
             irf=None,
             edisp=False,
             statistic='DEFAULT',
             components=False,
             stack=False,
             mask=False,
             algorithm='SIGNIFICANCE',
             logfile=None,
             silent=False):
    """
    Generates residual spectrum.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/csresspec.html

    Parameters
    ----------   
    - inobs (string): input observation file
    - inmodel (string): input model
    - outfile (str): output resdi spectrum file
    - npix (int): Number of cube bins
    - reso (float): Cube bin size (in degrees/pixel)
    - cra (float): Right Ascension of circular selection region centre (J2000, in degrees)
    - cdec (float): declination of circular selection region centre (J2000, in degrees)

    Outputs
    --------
    - create residual spectrum fits

    """

    if res_ra is None:
        res_ra  = cra
    if res_dec is None:
        res_dec = cdec
    
    rspec = cscripts.csresspec()

    rspec['inobs']      = inobs
    rspec['inmodel']    = inmodel
    if modcube   is not None: rspec['modcube']   = modcube
    if expcube   is not None: rspec['expcube']   = expcube
    if psfcube   is not None: rspec['psfcube']   = psfcube
    if edispcube is not None: rspec['edispcube'] = edispcube
    if bkgcube   is not None: rspec['bkgcube']   = bkgcube
    if caldb     is not None: rspec['caldb']     = caldb     
    if irf       is not None: rspec['irf']       = irf       
    rspec['edisp']      = edisp
    rspec['outfile']    = outfile
    rspec['statistic']  = statistic
    rspec['components'] = components
    rspec['ebinalg']    = ebinalg
    rspec['emin']       = emin
    rspec['emax']       = emax
    rspec['enumbins']   = enumbins
    rspec['ebinfile']   = 'NONE'
    rspec['stack']      = stack
    rspec['coordsys']   = 'CEL'
    rspec['proj']       = 'TAN'
    rspec['xref']       = cra
    rspec['yref']       = cdec
    rspec['nxpix']      = npix
    rspec['nypix']      = npix
    rspec['binsz']      = reso
    rspec['mask']       = mask
    rspec['ra']         = res_ra
    rspec['dec']        = res_dec
    rspec['rad']        = res_rad
    rspec['regfile']    = 'NONE'
    rspec['algorithm']  = algorithm
    if logfile is not None: rspec['logfile'] = logfile

    if logfile is not None: rspec.logFileOpen()
    rspec.execute()
    if logfile is not None: rspec.logFileClose()
    
    if not silent:
        print(rspec)
        print('')

    return rspec


