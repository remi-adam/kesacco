"""
This file contains the ObsSetup class. It is dedicated to the construction of an
object that defines the observational setup, which can be used in CTA simulations
or analysis.

"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import copy

from ClusterSimCTA.Common import background as model_bkg

import cscripts
import gammalib

#==================================================
# Total observation class
#==================================================

class ObsSetup(object):
    """ 
    ObsSetup class. 
    This class defines the ObsSetup object. The ObsSetup is given as 
    a list of observing run. It includes the background model for
    each run, as a Background object.

    Attributes
    ----------  
    - obsid (str): the unique observation ID
    - name (str): the label of the given observation run
    - coord (skycoord object): coordinates of the pointing
    - rad (quantity deg): radius of the pointing
    - tmin (str): time of observation start
    - tmax (str): time of observation end
    - emin (quantity energy): minimum recorded energy
    - emax (quantity energy): maximum recorded energy
    - edisp (Float): apply energy dispersion
    - deadc (float): Average deadtime correction factor
    - caldb (str): Calibration database
    - irf (str): Instrumental response function    
    - bkg (Background object): contain the background model
    - seed (int): the seed used for simulations of observations
    
    Methods
    ----------  
    - delete_obs: delete an observation run from the list
    - add_obs: add an observation run from the list
    - select_obs: select a given observation run an ObsSetup 
    that return an object that include only this run in the list
    - print_obs: print the observation runs
    
    """

    #==================================================
    # Initialize the object
    #==================================================

    def __init__(self):
        
        """
        Initialize the observation setup object.
        By default the list is empty.
        
        Parameters
        ----------

        """
        
        self.obsid = []
        self.name  = []
        self.coord = []
        self.rad   = []
        self.tmin  = []
        self.tmax  = []
        self.emin  = []
        self.emax  = []
        self.edisp = []
        self.deadc = []
        self.caldb = []
        self.irf   = []
        self.bkg   = []
        self.seed  = []
        
        
    #==================================================
    # Remove an observation run
    #==================================================

    def delete_obs(self, obsid):
        """
        Delete an observation from the list.
        
        Parameters
        ----------
        - obsid (str): name used for source label
        
        """

        w = np.where(np.array(self.obsid).astype('str') == obsid)[0]
        
        if len(w) > 1:
            raise ValueError("Unexpected error: the given name is matched to several existing observation.")
        
        if len(w) == 0:
            raise ValueError("The given observation does not exist.")
        
        if len(w) == 1:
            idx = w[0]
            del self.name[idx]
            del self.obsid[idx]
            del self.coord[idx]
            del self.rad[idx]
            del self.tmin[idx]
            del self.tmax[idx]
            del self.emin[idx]
            del self.emax[idx]
            del self.edisp[idx]
            del self.deadc[idx]
            del self.caldb[idx]
            del self.irf[idx]
            del self.bkg[idx]
            del self.seed[idx]
            
            
    #==================================================
    # Add an observation run
    #==================================================
    
    def add_obs(self,
                obsid='001',
                name='Ptg001',
                coord=SkyCoord(0*u.deg, 0*u.deg, frame='icrs'),
                rad=8*u.deg,
                tmin="2020-01-01T00:00:00.0",
                tmax="2020-01-01T01:00:00.0",
                emin=5e-2*u.TeV,
                emax=1e+2*u.TeV,
                edisp=False,
                deadc=0.95,
                caldb='prod3b-v2',
                irf='North_z20_S_5h',
                bkg=model_bkg.Background(),
                seed=None):
        """
        Add a new observation to the list.
        
        Parameters
        ----------
        - obsid (str): the unique observation ID
        - name (str): the label of the given observation run
        - coord (skycoord object): coordinates of the pointing
        - rad (quantity deg): radius of the pointing
        - tmin (str): time of observation start
        - tmax (str): time of observation end
        - emin (quantity energy): minimum recorded energy
        - emax (quantity energy): maximum recorded energy
        - edisp (Float): apply energy dispersion
        - deadc (float): Average deadtime correction factor
        - caldb (str): Calibration database
        - irf (str): Instrumental response function  
        - bkg (Background object): model of the background
        - seed (int): the seed used for simulations of observations
        """
        
        #----- Check the user parameters
        self._check_parameters(obsid, name,
                               coord, rad,
                               tmin, tmax,
                               emin, emax, edisp,
                               deadc, caldb, irf,
                               seed)
        
        #----- Add the observation
        w = np.where(np.array(self.obsid).astype('str') == obsid)[0]

        if len(w) > 0:
            print("!!!! WARNING !!!! The observation you are trying to add already exist.")
            print("                  Doing nothing.")
        else:
            self.obsid.append(obsid)
            self.name.append(name)
            self.coord.append(coord)
            self.rad.append(rad)
            self.tmin.append(tmin)
            self.tmax.append(tmax)
            self.emin.append(emin)
            self.emax.append(emax)
            self.edisp.append(edisp)
            self.deadc.append(deadc)
            self.caldb.append(caldb)
            self.irf.append(irf)
            self.bkg.append(bkg)
            self.seed.append(seed)
            

    #==================================================
    # Select an observation run
    #==================================================
    
    def select_obs(self, obsid):
        """
        Select an observing run and return the observing run 
        object as a list with a single element.
        
        Parameters
        ----------
        - obsid (str): the unique observation ID     
        """
        
        w = np.where(np.array(self.obsid).astype('str') == obsid)[0]

        if len(w) != 1:
            print("!!!! WARNING !!!! The observation you are trying to select does not exist.")
            print("                  Doing nothing.")
            return None
        
        else:
            idx = w[0]
            obj = copy.deepcopy(self)

            obj.obsid = [self.obsid[idx]]
            obj.name  = [self.name[idx]]
            obj.coord = [self.coord[idx]]
            obj.rad   = [self.rad[idx]]
            obj.tmin  = [self.tmin[idx]]
            obj.tmax  = [self.tmax[idx]]
            obj.emin  = [self.emin[idx]]
            obj.emax  = [self.emax[idx]]
            obj.edisp = [self.edisp[idx]]
            obj.deadc = [self.deadc[idx]]
            obj.caldb = [self.caldb[idx]]
            obj.irf   = [self.irf[idx]]
            obj.bkg   = [self.bkg[idx]]
            obj.seed  = [self.seed[idx]]

            return obj

        
    #==================================================
    # Get the minimal emin from the list
    #==================================================
    
    def get_emin(self):
        """
        Extract the minimal energy from the list
        
        Parameters
        ----------
        - emin (quantity): the minimal energy    
        """
        
        emin = np.zeros(len(self.obsid))*u.GeV
        for i in range(len(self.obsid)):
            emin[i] = self.emin[i]

        return np.amin(emin)


    #==================================================
    # Get the minimal emin from the list
    #==================================================
    
    def get_emax(self):
        """
        Extract the maximal energy from the list
        
        Parameters
        ----------
        - emax (quantity): the maximal energy    
        """
        
        emax = np.zeros(len(self.obsid))*u.GeV
        for i in range(len(self.obsid)):
            emax[i] = self.emax[i]

        return np.amax(emax)
    
    
    #==================================================
    # Print an observation
    #==================================================
        
    def print_obs(self):
        """
        Print the observations list.
            
        Parameters
        ----------
        
        """
        
        Nobs = len(self.obsid)

        if Nobs == 0:
            print('=== No observations setup are currently defined')
        
        for k in range(Nobs):
            print('=== '+self.name[k]+', ObsID '+self.obsid[k])
            print('        RA-Dec:    '+str(self.coord[k].icrs.ra.to_value('deg'))+', '+str(self.coord[k].icrs.dec.to_value('deg'))+' deg')
            print('        GLON-GLAT: '+str(self.coord[k].galactic.l.to_value('deg'))+', '+str(self.coord[k].galactic.b.to_value('deg'))+' deg')
            print('        ROI rad: '+str(self.rad[k].to_value('deg'))+' deg')
            print('        tmin: '+self.tmin[k])
            print('        tmax: '+self.tmax[k])
            print('        emin: '+str(self.emin[k].to_value('TeV'))+' TeV')
            print('        emax: '+str(self.emax[k].to_value('TeV'))+' TeV')
            print('        Apply Edisp: '+str(self.edisp[k]))
            print('        deadc: '+str(self.deadc[k]))
            print('        caldb: '+self.caldb[k]) 
            print('        irf: '+self.irf[k])
            print('        bkg: name '+self.bkg[k].name+', spatial type '+self.bkg[k].spatial['type']+', spectral type '+self.bkg[k].spectral['type'])
            print('        seed: '+str(self.seed[k]))


    #==================================================
    # Write pointing
    #==================================================
    
    def write_pnt(self, filename, obsid=None):
        """
        Write a pointing file.
        
        Parameters
        ----------
        - filename (str): the full name of the file to be written
        - obsid (str): the unique observation ID
        """

        f = open(filename, 'wb')
        f.write('name, id, ra, dec, duration, emin, emax, rad, deadc, caldb, irf \n')

        if obsid is not None:
            w = np.where(np.array(self.obsid).astype('str') == obsid)[0]
            if len(w) != 1: raise ValueError("The observation you are trying to write does not exist.")
            idx = w[0]

            f.write(self.name[idx]+', '
                    +self.obsid[idx] +', '
                    +str(self.coord[idx].icrs.ra.to_value('deg'))+', '
                    +str(self.coord[idx].icrs.dec.to_value('deg'))+', '
                    +str((Time(self.tmax[idx])-Time(self.tmin[idx])).sec)+', '
                    +str(self.emin[idx].to_value('TeV'))+', '
                    +str(self.emax[idx].to_value('TeV'))+', '
                    +str(self.rad[idx].to_value('deg'))+', '
                    +str(self.deadc[idx])+', '
                    +self.caldb[idx]+', '
                    +self.irf[idx]+' \n')            
        else:
            for idx in range(len(self.obsid)):
                f.write(self.name[idx]+', '
                        +self.obsid[idx] +', '
                        +str(self.coord[idx].icrs.ra.to_value('deg'))+', '
                        +str(self.coord[idx].icrs.dec.to_value('deg'))+', '
                        +str((Time(self.tmax[idx])-Time(self.tmin[idx])).sec)+', '
                        +str(self.emin[idx].to_value('TeV'))+', '
                        +str(self.emax[idx].to_value('TeV'))+', '
                        +str(self.rad[idx].to_value('deg'))+', '
                        +str(self.deadc[idx])+', '
                        +self.caldb[idx]+', '
                        +self.irf[idx]+' \n')
        f.close()


    #==================================================
    # Write pointing
    #==================================================
    
    def run_csobsdef(self, file_pnt, file_obsdef, silent=True):
        """
        Run the csobsdef script to get the observation definition file.
        
        Parameters
        ----------
        - file_pnt (str): the full name of the pointing file
        - file_obsdef (str): the full name of the file to be written
        - silent (bool): print the information
        """

        #----- Create base definition
        cobs = cscripts.csobsdef()
        cobs['inpnt']    = file_pnt
        cobs['outobs']   = file_obsdef
        cobs.execute()
        if not silent: print(cobs)

        #----- Add extra information
        #xml     = gammalib.GXml(file_obsdef)
        #obslist = xml.element('observation_list')
        #for idx in range(len(self.obsid)):
        #    obs = obslist[idx]
        #    obs.append('parameter name="EventList" file="'+os.path.dirname(file_obsdef)+'/EVENTS'+self.obsid[idx]+'.fits"')
        #xml.save(file_obsdef)        
        
        
    #==================================================
    # Check that the parameters are ok
    #==================================================
        
    def _check_parameters(self, obsid, name, coord, rad, tmin, tmax, emin, emax, edisp, deadc, caldb, irf, seed):
        """
        Check that the parameters are ok in terms of type and values
            
        Parameters
        ----------
        - obsid (str): the unique observation ID
        - name (str): the label of the given observation run
        - coord (skycoord object): coordinates of the pointing
        - rad (quantity deg): radius of the pointing
        - tmin (str): time of observation start
        - tmax (str): time of observation end
        - emin (quantity energy): minimum recorded energy
        - emax (quantity energy): maximum recorded energy
        - edisp (Float): apply energy dispersion
        - deadc (float): Average deadtime correction factor
        - caldb (str): Calibration database
        - irf (str): Instrumental response function
        """

        #----- obsid
        if type(obsid) != str:
            raise TypeError("The parameter 'obsid' should be a string ")

        #----- Name
        if type(name) != str:
            raise TypeError("The parameter 'name' should be a string ")
        
        #----- coord
        if type(coord) != SkyCoord:
            raise TypeError("The parameter 'coord' should be a skycoord object")
        
        #----- rad
        try:
            test = rad.to('deg')
        except:
            raise TypeError("The parameter 'rad' should be a quantity homogeneous to deg")
        if rad.value <= 0:
            raise TypeError("The parameter 'rad' should be larger than 0")

        #----- tmin/tmax
        if type(tmin) != str or type(tmax) != str:
            raise TypeError("The parameter 'tmin,tmax' should be a string, e.g. 2020-01-01T00:00:00.0")
        dt = Time(tmax) - Time(tmin)
        if dt.value <=0:
            raise TypeError("The parameter 'tmin' should be lower than 'tmax'")
        
        #----- emin/emax
        try:
            test = emin.to('TeV')
            test = emax.to('TeV')
        except:
            raise TypeError("The parameter 'emin,emax' should be a quantity homogeneous to TeV")
        if emin >= emax:
            raise TypeError("The parameter 'emin' should be lower than 'emax'")

        #----- edisp
        if type(edisp) != bool:
            raise TypeError("The parameter 'edisp' should be a boolean")

        #----- deadc
        if type(deadc) != float and type(deadc) != int and type(deadc) != np.float64:
            raise TypeError("The parameter 'deadc' should be a float")
                            
        if deadc <= 0 or deadc > 1:
            raise TypeError("The parameter 'deadc' should be between 0 and 1")

        #----- caldb
        if type(caldb) != str:        
            raise TypeError("The parameter 'caldb' should be a string")

        #----- irf
        if type(irf) != str:        
            raise TypeError("The parameter 'irf' should be a string")
        
        #----- seed
        if seed is not None and type(seed) != int:
            raise TypeError("The parameter 'seed' should be a int")
