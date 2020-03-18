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

    spec.execute()

    if not silent:
        print(spec)
        
    return spec


#==================================================
# Butterfly
#==================================================
def butterfly():
    """
    Computes butterfly diagram for a given spectral model.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/ctbutterfly.html

    Parameters
    ----------
    - 
    -
    -
    -

    Outputs
    --------
    - create butterfly fits

    """
    
    but = ctools.ctbutterfly()

    '''
            #========== Binned analysis     
            if param['binned']:
                but['inobs']     = param['output_dir']+'/clustana_DataPrep_ctbin.fits'
                but['inmodel']   = param['output_dir']+'/clustana_output_model.xml'
                but['expcube']   = param['output_dir']+'/clustana_DataPrep_ctexpcube.fits'
                but['psfcube']   = param['output_dir']+'/clustana_DataPrep_ctpsfcube.fits'
                if param["apply_edisp"]:
                    but['edispcube'] = param['output_dir']+'/clustana_DataPrep_ctedispcube.fits'
                if param['bkg_spec_Prefactor'] > 0:
                    but['bkgcube']   = param['output_dir']+'/clustana_DataPrep_ctbkgcube.fits'
                else: 
                    but['bkgcube']   = 'NONE'
            
            #========== Unbinned analysis  
            else:
                but['inobs']     = param['output_dir']+'/clustana_DataPrep_ctselect.fits'
                but['inmodel']   = param['output_dir']+'/clustana_output_model.xml'

            #========== General param and run
            but['srcname']    = isource_name
            but['caldb']      = param['caldb']
            but['irf']        = param['irf']
            but['edisp']      = param['apply_edisp']
            but['outfile']    = param['output_dir']+'/clustana_spectrumb_'+isource_name+'.txt'
            but['fit']        = False # To refit
            but['method']     = 'GAUSSIAN'
            but['confidence'] = 0.68
            but['statistic']  = 'DEFAULT'
            but['matrix']     = 'NONE' #param['output_dir']+'/clustana_likelihood_covmat.fits'
            but['ebinalg']    = param['ebinalg']
            but['emin']       = param['emin'].to_value('TeV') * 0.5
            but['emax']       = param['emax'].to_value('TeV') * 2.0
            but['enumbins']   = 100
    '''

    but.execute()

    if not silent:
        print(but)

    return but




#==================================================
# Spectrum residual
#==================================================
def residual():
    """
    Generates residual spectrum.
    See http://cta.irap.omp.eu/ctools/users/reference_manual/csresspec.html

    Parameters
    ----------
    -
    -
    -
    -

    Outputs
    --------
    - create residual spectrum fits

    """
    
    rspec = cscripts.csresspec()

    '''
    #========== Binned analysis     
    if param['binned']:
        resi['inobs']     = param['output_dir']+'/clustana_DataPrep_ctbin.fits'
        resi['inmodel']   = param['output_dir']+'/clustana_output_model.xml'
        resi['expcube']   = param['output_dir']+'/clustana_DataPrep_ctexpcube.fits'
        resi['psfcube']   = param['output_dir']+'/clustana_DataPrep_ctpsfcube.fits'
        if param["apply_edisp"]:
            resi['edispcube'] = param['output_dir']+'/clustana_DataPrep_ctedispcube.fits'
        if param['bkg_spec_Prefactor'] > 0:
            resi['bkgcube']   = param['output_dir']+'/clustana_DataPrep_ctbkgcube.fits'
        else: 
            resi['bkgcube']   = 'NONE'
            
    #========== Unbinned analysis  
    else:
        resi['inobs']     = param['output_dir']+'/clustana_DataPrep_ctselect.fits'
        resi['inmodel']   = param['output_dir']+'/clustana_output_model.xml'

    #========== General param and run
    resi['caldb']      = param['caldb']
    resi['irf']        = param['irf']
    resi['edisp']      = param['apply_edisp']
    resi['outfile']    = param['output_dir']+'/clustana_spectrumr.fits'
    resi['statistic']  = 'DEFAULT'
    resi['components'] = True
    resi['ebinalg']    = param['ebinalg']
    resi['emin']       = param['emin'].to_value('TeV')
    resi['emax']       = param['emax'].to_value('TeV')
    resi['enumbins']   = param['enumbins']
    resi['mask']       = False
    resi['ra']         = param['cluster_ra'].to_value('deg')
    resi['dec']        = param['cluster_dec'].to_value('deg')
    resi['rad']        = param['cluster_t500'].to_value('deg')
    resi['algorithm']  = 'SIGNIFICANCE'

    '''
    
    rspec.execute()

    if not silent:
        print(rspec)

    return rspec


