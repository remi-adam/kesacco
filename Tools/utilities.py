"""
This script gather various function that are used as 
utilities in various modules
"""

#==================================================
# Requested imports
#==================================================
from astropy.units.quantity import Quantity as Qtype
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.coordinates import cartesian_to_spherical
import astropy.units as u
from scipy.optimize import minimize
import numpy as np

#==================================================
# get npix from FoV definition
#==================================================

def npix_from_fov_def(fov, reso):
    """
    Return the number of pixels given a field of view size and 
    the pixel size.
    
    Parameters
    ----------
    - FoV (float): the field of view size
    - reso (float): the pixel size

    Outputs
    --------
    - npix (int): the number of pixels along the axis
    """

    # Make sure units are fine
    if type(fov) == Qtype and type(reso) != Qtype:
        raise TypeError('fov and reso should both be quantities homogeneous to deg, or both scalars.')

    if type(fov) != Qtype and type(reso) == Qtype:
        raise TypeError('fov and reso should both be quantities homogeneous to deg, or both scalars.')
    
    try:
        npix = int(fov.to_value('deg')/reso.to_value('deg'))
    except:
        npix = int(fov/reso)

    # Make sure to have one pixel at the center
    if npix/2.0 == int(npix/2.0): npix += 1

    return npix


#==================================================
# get the center of a list of coordinates
#==================================================

def listcord2center(coord_list):
    """
    Return the center of a list of coordinates. This is 
    done by averaging cartesian coordinates to avoid 
    looping in RA-Dec.
    
    Parameters
    ----------
    - coord_list (SkyCoord list): list of sky coordinates

    Outputs
    --------
    - center (SkyCoord): SkyCoord object, center of the list
    """
    
    # Get the cartesian coordinates
    x = coord_list.cartesian.x
    y = coord_list.cartesian.y
    z = coord_list.cartesian.z

    # Average the cartesian coordinates
    x_m = np.mean(x)
    y_m = np.mean(y)
    z_m = np.mean(z)

    # Transform to sky coordinates
    r, lat, lon = cartesian_to_spherical(x_m, y_m, z_m)
    center_guess = SkyCoord(lon, lat, frame='icrs')

    # Perform distance minimisation to make sure it is ok
    def fun(par):
        c = SkyCoord(par[0]*u.deg, par[1]*u.deg, frame='icrs')
        dist = coord_list.separation(c)
        return np.sum(dist.value**2)

    p_guess = np.array([center_guess.ra.to_value('deg'), center_guess.dec.to_value('deg')])
    res = minimize(fun, p_guess)
    center = SkyCoord(res.x[0]*u.deg, res.x[1]*u.deg, frame='icrs')

    # Sanity check
    if res.success is not True:
        print('!!! WARNING: not sure I found the coordinates barycenter !!!')
        print('Separation between cartesian center and my center: ')
        print(center.separation(center_guess).to_value('deg'))
    
    return center


#==================================================
# get the size of a required field of view
#==================================================

def listcord2fov(coord_list, rad_list):
    """
    Return the center and the size of a field of view that 
    would enclose a list of pointing (centers + radius).
    
    Parameters
    ----------
    - coord_list (SkyCoord list): list of sky coordinates
    - rad_list (deg): list of radius associated to center
    
    Outputs
    --------
    - center (SkyCoord): SkyCoord object
    """

    # Compute the overall center
    center = listcord2center(coord_list)

    # Get the distance from all pointing plus extension
    sep  = coord_list.separation(center)

    # Add the extenstion
    dist = np.zeros(len(sep))
    for i in range(len(sep)):
        dist[i] = sep[i].to_value('deg') + rad_list[i].to_value('deg')

    fov = 2*np.amax(dist)*u.deg

    return center, fov


#==================================================
# get size of field of view including cluster
#==================================================

def squeeze_fov(center_fov, fov_ini, center_cluster, theta_cluster, extra=1.1):
    """
    Take a map center and fov definition, and squeeze it 
    so that the cluster inside remains entirely inside 
    the map. In case the cluster is already bigger than 
    the map, it should do nothing.
    
    Parameters
    ----------
    - center_fov (SkyCoord): center of the field of view
    - fov_ini (quantity, deg): initial FoV size
    - center_cluster (SkyCoord): center of the cluster
    - theta_cluster (quantity, deg): cluster max extent
    
    Outputs
    --------
    - fov (quantity, deg): size of the FoV
    """
    
    # Distance along RA between cluster center and map center
    sep_x = SkyCoord(center_cluster.icrs.ra, center_fov.icrs.dec, frame='icrs').separation(center_fov)
    # Distance along Dec between cluster center and map center
    sep_y = SkyCoord(center_fov.icrs.ra, center_cluster.icrs.dec, frame='icrs').separation(center_fov)

    # Compute FoV along x
    fov_x = 2*np.amin([fov_ini.to_value('deg')/2.0,
                       (sep_x+extra*theta_cluster).to_value('deg')])

    # Compute FoV along y
    fov_y = 2*np.amin([fov_ini.to_value('deg')/2.0,
                       (sep_x+extra*theta_cluster).to_value('deg')])

    # The fov is squared, so take the max
    fov = np.amax([fov_x, fov_y])*u.deg

    return fov



