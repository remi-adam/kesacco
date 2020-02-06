"""
This file contains parser to ctools for scripts related to 
binning the data.
"""

#==================================================
# Imports
#==================================================

import ctools
import numpy as np


#==================================================
# Counts bin
#==================================================

def counts_cube():
    """
    Compute counts cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbin.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    binning = ctools.ctbin()

    binning.execute()
    
    return binning


#==================================================
# Exposure
#==================================================

def exp_cube():
    """
    Compute a exposure cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctexpcube.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    expcube = ctools.ctexpcube()

    expcube.execute()
    
    return expcube


#==================================================
# PSF cube
#==================================================

def psf_cube():
    """
    Compute a PSF cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctpsfcube.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    psfcube = ctools.ctpsfcube()

    psfcube.execute()

    return psfcube


#==================================================
# Bkg cube
#==================================================

def bkg_cube():
    """
    Compute a background cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctbkgcube.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    bkgcube = ctools.ctbkgcube()

    bkgcube.execute()
    
    return bkgcube


#==================================================
# Edisp cube
#==================================================

def edisp_cube():
    """
    Compute an energy dispersion cube.
    http://cta.irap.omp.eu/ctools/users/reference_manual/ctedispcube.html

    Parameters
    ----------

    Outputs
    --------
    """
    
    edc= ctools.ctedispcube()

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
    
    maskcube = ctools.ctcubemask()

    maskcube.execute()
    
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
