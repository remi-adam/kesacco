"""
This file contains functions dedicated to the creation of gamma ray templates
(spectrum and maps) of galaxy clusters, which can be used in ctools. The 
templates are built from a Cluster object (Model class from ClusterModel)

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from astropy.io import fits
import astropy.units as u


#==================================================
# Maps
#==================================================

def make_map(cluster,
             filename,
             Egmin=5e-2*u.TeV,
             Egmax=1e+2*u.TeV,
             includeIC=False):
    """
    Compute the map of a cluster for ctools.
        
    Parameters
    ----------
    - cluster: ClusterModel object
    - filename: in which file to save the results
    - Egmin/Egmax (quantity energy): Egmin/Egmax define the energy range 
    used to compute the template maps. This has no effect on pion decay
    emission because the shape of the profile is the same at all energies,
    but has (very) little effect on IC emission.
    - includeIC (bool): include inverse compton emission or not
    
    Outputs
    --------
    - fits file map is saved
    """
    
    header = cluster.get_map_header()
    
    #---------- pion decay
    image = cluster.get_gamma_map(Emin=Egmin, Emax=Egmax,
                                  Rmin_los=None, NR500_los=5.0,
                                  Rmin=None, Rmax=None,
                                  Normalize=True)
    
    #---------- IC
    if includeIC:
        image += cluster.get_ic_map(Emin=Egmin, Emax=Egmax,
                                    Rmin_los=None, NR500_los=5.0,
                                    Rmin=None, Rmax=None,
                                    Normalize=True)
        print('!!!!! WARNING: including the IC contribution will lead to non normalized map right now !!!!!')
    
    #---------- Write the fits
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = image.value
    hdu.header.add_comment('Gamma map')
    hdu.header.add_comment('Unit = '+str(image.unit))
    hdu.writeto(filename, overwrite=True)
    
    
#==================================================
# Spectrum
#==================================================

def make_spectrum(cluster,
                  filename,
                  energy=np.logspace(-2,6,1000)*u.GeV,
                  includeIC=False):
    """
    Compute the spectrum of a cluster for ctools.
    
    Parameters
    ----------
    - cluster: ClusterModel object
    - filename: in which file to save the results
    - energy (quantity array): the photon energy sampling
    - includeIC (bool): include inverse compton emission or not
    
    Outputs
    --------
    - fits file map is saved
    """
    
    #---------- pion decay
    eng, spec = cluster.get_gamma_spectrum(energy,
                                           Rmin=None, Rmax=cluster.R_truncation,
                                           Rmin_los=None, NR500_los=5.0,
                                           type_integral='spherical')
    
    #---------- IC
    if includeIC:
        eng_ic, spec_ic = cluster.get_ic_spectrum(energy,
                                                  Rmin=None, Rmax=cluster.R_truncation,
                                                  Rmin_los=None, NR500_los=5.0,
                                                  type_integral='spherical')
        print('!!!!! WARNING: including the IC contribution will lead to non normalized map right now !!!!!')
        spec += spec_ic

    #---------- Remove zero from the spectrum
    wgood  = spec >0
    energy = energy[wgood]
    spec   = spec[wgood]

    #---------- Write the file
    cluster._save_txt_file(filename,
                           energy.to_value('MeV'),
                           spec.to_value('MeV-1 cm-2 s-1'),
                           'energy (MeV)',
                           'spectrum (MeV-1 cm-2 s-1)')
        
    
