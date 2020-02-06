"""
This file contains functions dedicated to the creation of as ctools
like model file.

"""

#==================================================
# Requested imports
#==================================================

import astropy.units as u
import gammalib


#==================================================
# Cluster
#==================================================

def cluster(model_tot, file_map, file_spec, ClusterName='Cluster'):
    """
    Build a ctools model for the cluster.
        
    Parameters
    ----------
    - model_tot: a gammalib.GModels object
    - file_map (str): the full name of the fits map
    - file_spec (str): the full name of the text file spectrum
    
    Outputs
    --------
    - the model is updated to include the cluster
    """

    spatial  = gammalib.GModelSpatialDiffuseMap(file_map)
    spectral = gammalib.GModelSpectralFunc(file_spec, 1.0)
    timing   = gammalib.GModelTemporalConst(1)
    
    model    = gammalib.GModelSky(spatial, spectral, timing)
    model.name(ClusterName)

    model_tot.append(model)
    
    
#==================================================
# Point sources
#==================================================

def compact_sources(model_tot, source_dict):
    """
    Build a ctools model for the compact sources.
    
    Parameters
    ----------
    - model_tot: a gammalib.GModels object
    - source_dict (dictionary): the source dictionary 
    as defined by the CompactSource class
    
    Outputs
    --------
    - the model is updated to include the sources
    """
    
    Nsource = len(source_dict.name)
    
    for ips in range(Nsource):
        #----- Spatial model
        # PointSource
        if source_dict.spatial[ips]['type'] == 'PointSource':
            RA  = source_dict.spatial[ips]['param']['RA']['value'].to_value('deg')
            Dec = source_dict.spatial[ips]['param']['DEC']['value'].to_value('deg')
            spatial = gammalib.GModelSpatialPointSource(RA, Dec)

        # Error
        else:
            raise ValueError('Spatial model not avaiable')

        #----- Spectral model
        # PowerLaw
        if source_dict.spectral[ips]['type'] == 'PowerLaw':
            prefact = source_dict.spectral[ips]['param']['Prefactor']['value'].to_value('cm-2 s-1 MeV-1')
            index   = source_dict.spectral[ips]['param']['Index']['value']
            pivot   = gammalib.GEnergy(source_dict.spectral[ips]['param']['PivotEnergy']['value'].to_value('TeV'), 'TeV')
	    spectral = gammalib.GModelSpectralPlaw(prefact, index, pivot)

        # PowerLawExpCutoff
        elif source_dict.spectral[ips]['type'] == 'PowerLawExpCutoff':
            prefact = source_dict.spectral[ips]['param']['Prefactor']['value'].to_value('cm-2 s-1 MeV-1')
            index   = source_dict.spectral[ips]['param']['Index']['value']
            pivot   = gammalib.GEnergy(source_dict.spectral[ips]['param']['PivotEnergy']['value'].to_value('TeV'), 'TeV')
            cutoff  = gammalib.GEnergy(source_dict.spectral[ips]['param']['Cutoff']['value'].to_value('TeV'), 'TeV')
	    spectral = gammalib.GModelSpectralExpPlaw(prefact, index, pivot, cutoff)

        # Error
        else:
            raise ValueError('Spectral model not available')

	#----- Temporal model
        # Constant
        if source_dict.temporal[ips]['type'] == 'Constant':
	    temporal = gammalib.GModelTemporalConst(source_dict.temporal[ips]['param']['Normalization']['value'])

        # Error
        else:
            raise ValueError('Temporal model not available')

        #----- Parameter management
        spatial  = manage_parameters(source_dict.spatial[ips]['param'], spatial)
        spectral = manage_parameters(source_dict.spectral[ips]['param'], spectral)
        temporal = manage_parameters(source_dict.temporal[ips]['param'], temporal)

	#----- Overal model for each source
	model = gammalib.GModelSky(spatial, spectral, temporal)
	model.name(source_dict.name[ips])
	
	#----- Append model for each source
	model_tot.append(model)
    
    
#==================================================
# Background
#==================================================

def background(model_tot, bkg_dict_in, setID=True):
    """
    Build a ctools model for the background.
    
    Parameters
    ----------
    - model_tot: a gammalib.GModels object
    - bkg_dict (dictionary): the dictionary that 
    contain the background properties (from class 
    Background). In case of multiple background this can be a list.
    - setID (bool): decide to set or not an ID in the bkg model
    
    Outputs
    --------
    - the model is updated to include the background
    """

    #---------- Get the number of background
    if type(bkg_dict_in) == list:
        Nbkg = len(bkg_dict_in)
    else:
        Nbkg = 1

    #---------- Add all bkg models
    for i in range(Nbkg):
        #---------- Select the background from the list or not
        if type(bkg_dict_in) == list:
            bkg_dict = bkg_dict_in[i]
        else:
            bkg_dict = bkg_dict_in

        #----- Spectral model
        
        # PowerLaw
        if bkg_dict.spectral['type'] == 'PowerLaw':
            prefact = bkg_dict.spectral['param']['Prefactor']['value']
            index   = bkg_dict.spectral['param']['Index']['value']
            pivot   = gammalib.GEnergy(bkg_dict.spectral['param']['PivotEnergy']['value'].to_value('TeV'), 'TeV')
	    spectral = gammalib.GModelSpectralPlaw(prefact, index, pivot)
        
        # PowerLawExpCutoff
        elif bkg_dict.spectral['type'] == 'PowerLawExpCutoff':
            prefact = bkg_dict.spectral['param']['Prefactor']['value']
            index   = bkg_dict.spectral['param']['Index']['value']
            pivot   = gammalib.GEnergy(bkg_dict.spectral['param']['PivotEnergy']['value'].to_value('TeV'), 'TeV')
            cutoff   = gammalib.GEnergy(bkg_dict.spectral['param']['Cutoff']['value'].to_value('TeV'), 'TeV')
	    spectral = gammalib.GModelSpectralExpPlaw(prefact, index, pivot, cutoff)
        
        # Error
        else:
            raise ValueError('Spectral model not available')

        # Parameter management
        spectral = manage_parameters(bkg_dict.spectral['param'], spectral)
    
        #----- Spatial model
        # CTAIrfBackground
        if bkg_dict.spatial['type'] == 'CTAIrfBackground':

            #----- Overal model for each source
            model = gammalib.GCTAModelIrfBackground(spectral)
        
        # Error
        else:
            raise ValueError('Spatial model not avaiable')
    
        #----- Append model
        model.name(bkg_dict.name)
        model.instruments(bkg_dict.instrument)
        if setID:
            if bkg_dict.obsid is not None: model.ids(bkg_dict.obsid)
    
        model_tot.append(model)


#==================================================
# Manage parameters
#==================================================

def manage_parameters(params, model):
    """
    Manage the properties of parameters.
    
    Parameters
    ----------
    - params: the disctionary of parameters
    - model: the model to manage
    
    Outputs
    --------
    - model: including parameters constraints
    """

    for key in params:
        # Check for free keyword
        if 'free' in params[key]:
            if params[key]['free']:
                model[key].free()
            else:            
                model[key].fix()

        # Check for min keyword
        if 'min' in params[key]:
            model[key].min(params[keys]['min'])
        
        # Check for max keyword
        if 'max' in params[key]:
            model[key].min(params[keys]['max'])
        
        # Check for error keyword
        if 'error' in params[key]:
            model[key].error(params[keys]['error'])
            
        # Check for scale keyword
        if 'scale' in params[key]:
            model[key].error(params[keys]['scale'])
            
    return model
