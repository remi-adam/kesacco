"""
This file contains parser to ctools for scripts related
to spectra.
"""

import gammalib
import ctools
import cscripts

def main(param):
    """
    Perform spectrum analysis

    Parameters
    ----------
    - param: dictionary parameter file
    
    Outputs
    --------
    - spectra files
    """
    
    print('')
    print('************************************************')
    print('-------------- Compute spectra -----------------')
    print('************************************************')
    
    models = gammalib.GModels(param['output_dir']+'/clustana_input_model.xml')
    nsource = len(models)

    #******************************************************************************************
    #------------------------------------- Compute spectra ------------------------------------
    #******************************************************************************************
    
    for isource in range(nsource):
        
        isource_name = models[isource].name()

        spec = cscripts.csspec()

        #========== Binned analysis     
        if param['binned']:
            spec['inobs']     = param['output_dir']+'/clustana_DataPrep_ctbin.fits'
            spec['expcube']   = param['output_dir']+'/clustana_DataPrep_ctexpcube.fits'
            spec['psfcube']   = param['output_dir']+'/clustana_DataPrep_ctpsfcube.fits'
            if param["apply_edisp"]:
                spec['edispcube'] = param['output_dir']+'/clustana_DataPrep_ctedispcube.fits'
            if param['bkg_spec_Prefactor'] > 0:
                spec['bkgcube']   = param['output_dir']+'/clustana_DataPrep_ctbkgcube.fits'
            else: 
                spec['bkgcube']   = 'NONE'
        
        #========== Unbinned analysis 
        else:
            spec['inobs']     = param['output_dir']+'/clustana_DataPrep_ctselect.fits'
            
        #========== General param and run
        spec['inmodel']   = param['output_dir']+'/clustana_output_model.xml'
        spec['srcname']   = isource_name
        spec['caldb']     = param['caldb']
        spec['irf']       = param['irf']
        spec['edisp']     = param['apply_edisp']
        spec['outfile']   = param['output_dir']+'/clustana_spectrum_'+isource_name+'.fits'
        spec['method']    = 'SLICE'
        spec['ebinalg']   = param['ebinalg']
        spec['emin']      = param['emin'].to_value('TeV')
        spec['emax']      = param['emax'].to_value('TeV')
        spec['enumbins']  = param['enumbins']
        spec['statistic'] = 'DEFAULT'
        spec['calc_ts']   = True
        spec['calc_ulim'] = True
        spec['fix_srcs']  = False
        spec['fix_bkg']   = False
      
        spec.run()
        spec.save()
        print('---------- '+isource_name+' ----------')
        print(spec)
        print('')

    #******************************************************************************************
    #------------------------------------- Compute Residual -----------------------------------
    #******************************************************************************************
    resi = cscripts.csresspec()
    
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

    resi.execute()
    print('---------- Residual ----------')
    print(resi)
    print('')

    #******************************************************************************************
    #------------------------------------- Compute Butterfly ----------------------------------
    #******************************************************************************************

    for isource in range(nsource):
       
        isource_name = models[isource].name()

        if isource_name != 'Background':
            but = ctools.ctbutterfly()
    
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
        
            but.execute()
            print('---------- Butterfly: '+isource_name+'----------')
            print(but)
            print('')
