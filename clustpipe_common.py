"""
This file contains the Common class, which list 
functions used for common (e.g. administrative) tasks 
related to the ClusterPipe.
"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import pickle
import gammalib
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

from ClusterPipe.Tools import make_cluster_template
from ClusterPipe.Tools import build_ctools_model
from ClusterPipe.Tools import utilities

#==================================================
# Cluster class
#==================================================

class Common():
    """ 
    Admin class. 
    This class serves as parser to the ClusterPipe class and 
    contains administrative related tools.
    
    Methods
    ----------  
    - config_save(self, config_file)
    - config_load(self)
    - _check_obsID(self, obsID)
    - _correct_eventfile_names(self, xmlfile, prefix='Events')
    - _write_new_xmlevent_from_obsid(self, xmlin, xmlout, obsID)
    - _rm_source_xml(self, xmlin, xmlout, source)
    - _match_cluster_to_pointing(self, extra=1.1)
    - _match_anamap_to_pointing(self, extra=1.1)
    - _make_model(self, prefix='Model', includeIC=True)

    """
        
    #==================================================
    # Save the simulation configuration
    #==================================================
    
    def config_save(self):
        """
        Save the configuration for latter use
        
        Parameters
        ----------

        Outputs
        -------
        
        """
        
        # Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        # Save
        with open(self.output_dir+'/config.pkl', 'wb') as pfile:
            pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)
            
            
    #==================================================
    # Load the simulation configuration
    #==================================================
    
    def config_load(self, config_file):
        """
        Save the configuration for latter use
        
        Parameters
        ----------
        - config_file (str): the full name to the configuration file

        Outputs
        -------
        
        """

        with open(config_file, 'rb') as pfile:
            par = pickle.load(pfile)
            
        self.__dict__ = par


    #==================================================
    # Check obsID
    #==================================================
    
    def _check_obsID(self, obsID):
        """
        Validate the obsID given by the user
        
        Parameters
        ----------
        - obsID
        
        Outputs
        -------
        - obsID (list): obsID once validated
        
        """
        
        if obsID is None:
            obsID = self.obs_setup.obsid
            
        else:        
            # Case of obsID as a string, i.e. single run
            if type(obsID) == str:
                if obsID in self.obs_setup.obsid:
                    obsID = [obsID] # all good, just make it as a list
                else:
                    raise ValueError("The given obsID does not match any of the available observation ID.")
                
            # Case of obsID as a list, i.e. multiple run
            elif type(obsID) == list:
                good = np.array(obsID) == np.array(obsID) # create an array of True
                for i in range(len(obsID)):
                    # Check type
                    if type(obsID[i]) != str:
                        raise ValueError("The given obsID should be a string or a list of string.")
                
                    # Check if the obsID is valid and flag
                    if obsID[i] not in self.obs_setup.obsid:
                        if not self.silent: print('WARNING: obsID '+obsID[i]+' does not exist, ignore it')
                        good[i] = False

                # Select valid obsID
                if np.sum(good) == 0:
                    raise ValueError("None of the given obsID exist")
                else:
                    obsID = list(np.array(obsID)[good])
                    
                # Remove duplicate
                obsID = list(set(obsID))
                    
                # Case of from format
            else:
                raise ValueError("The obsID should be either a list or a string.")
    
        return obsID


    #==================================================
    # Correct XML and file names for simulated events
    #==================================================
    
    def _correct_eventfile_names(self, xmlfile, prefix='Events'):
        """
        Change the event filename and associated xml file
        by naming them using the obsid
        
        Parameters
        ----------
        - xmlfile (str): the xml file name
        
        """
        
        xml     = gammalib.GXml(xmlfile)
        obslist = xml.element('observation_list')
        for i in range(len(obslist)):
            obs = obslist[i]
            obsid = obs.attribute('id')

            # make sure the EventList key exist
            killnum = None
            for j in range(len(obs)):
                if obs[j].attribute('name') == 'EventList': killnum = j

            # In case there is one EventList, move the file and rename the xml
            if killnum is not None:
                file_in = obs[killnum].attribute('file')
                file_out = os.path.dirname(file_in)+'/'+prefix+obsid+'.fits'
                os.rename(file_in, file_out)
                obs.remove(killnum)
                obs.append('parameter name="EventList" file="'+file_out+'"')
                
        xml.save(xmlfile)      

        
    #==================================================
    # Write new events.xml file based on obsID
    #==================================================
    
    def _write_new_xmlevent_from_obsid(self, xmlin, xmlout, obsID):
        """
        Read the xml file gathering the event list, remove the event
        files that are not selected, and write a new xml file.
        
        Parameters
        ----------
        - xmlin (str): input xml file
        - xmlout (str): output xml file
        - obsID (list): list of str

        Outputs
        -------
        - The new xml file is writen
        """

        xml     = gammalib.GXml(xmlin)
        obslist = xml.element('observation_list')
        obsid_in = []
        for i in range(len(obslist))[::-1]:
            if obslist[i].attribute('id') not in obsID:
                obslist.remove(i)
            else:
                obsid_in.append(obslist[i].attribute('id'))
        for i in range(len(obsID)):
            if obsID[i] not in obsid_in:
                print('WARNING: Event file with obsID '+obsID[i]+' does not exist. It is ignored.')
        xml.save(xmlout)


    #==================================================
    # Remove a given source from a xml model
    #==================================================
    
    def _rm_source_xml(self, xmlin, xmlout, source):
        """
        Read a model xml file, remove a given source, and write 
        a new model file.
        
        Parameters
        ----------
        - xmlin (str): input xml file
        - xmlout (str): output xml file
        - source (str): name of the source to remove

        Outputs
        -------
        - The new xml file is writen
        """

        xml = gammalib.GXml(xmlin)
        srclist = xml.element('source_library')
        for i in range(len(srclist))[::-1]:
            if srclist[i].attribute('name') == source: 
                srclist.remove(i)
        xml.save(xmlout)
        
        
    #==================================================
    # Match the cluster map and FoV to the pointing def
    #==================================================
    
    def _match_cluster_to_pointing(self, extra=1.1):
        """
        Match the cluster map and according to the pointing list.
        
        Parameters
        ----------
        - extra (float): factor to apply to the cluster extent to 
        have a bit of margin on the side of the map

        Outputs
        -------
        The cluster map properties are modified
        
        """

        # Compute the pointing barycenter and the FoV requested size 
        list_ptg_coord = SkyCoord(self.obs_setup.coord)
        list_ptg_rad   = self.obs_setup.rad
        center_ptg, fov_ptg = utilities.listcord2fov(list_ptg_coord, list_ptg_rad)

        # Account for the cluster
        fov = utilities.squeeze_fov(center_ptg, fov_ptg,
                                    self.cluster.coord, self.cluster.theta_truncation,
                                    extra=extra)

        # Set the cluster map to match the pointing
        self.cluster.map_coord = center_ptg
        self.cluster.map_fov   = fov

        
    #==================================================
    # Match clustpipe map and FoV to the pointing def
    #==================================================
    
    def _match_anamap_to_pointing(self, extra=1.1):
        """
        Match the ClusterPipe map and according to the pointing list.
        
        Parameters
        ----------
        - extra (float): factor to apply to the cluster extent to 
        have a bit of margin on the side of the map

        Outputs
        -------
        The ClusterPipe map properties are modified
        
        """

        # Compute the pointing barycenter and the FoV requested size 
        list_ptg_coord = SkyCoord(self.obs_setup.coord)
        list_ptg_rad   = self.obs_setup.rad
        center_ptg, fov_ptg = utilities.listcord2fov(list_ptg_coord, list_ptg_rad)

        # Account for the cluster
        fov = utilities.squeeze_fov(center_ptg, fov_ptg,
                                    self.cluster.coord, self.cluster.theta_truncation,
                                    extra=extra)

        # Set the cluster map to match the pointing
        self.map_coord = center_ptg
        self.map_fov   = fov

        
    #==================================================
    # Make a model
    #==================================================
    
    def _make_model(self, prefix='Model', includeIC=True, obsID=None):
        """
        This function is used to construct the model.
        
        Parameters
        ----------
        - prefix (str): text to add as a prefix of the file names
        - includeIC (bool): include inverse Compton in the model
        
        """
        
        #----- Make cluster template files
        if (not self.silent) and (self.cluster.map_reso > 0.02*u.deg):
            print('WARNING: the FITS map resolution (self.cluster.map_reso) ')
            print('         is larger than 0.02 deg, i.e. the PSF at 100 TeV')
        
        make_cluster_template.make_map(self.cluster,
                                       self.output_dir+'/'+prefix+'_Map.fits',
                                       Egmin=self.obs_setup.get_emin(),
                                       Egmax=self.obs_setup.get_emax(),
                                       includeIC=includeIC)
        
        make_cluster_template.make_spectrum(self.cluster,
                                            self.output_dir+'/'+prefix+'_Spectrum.txt',
                                            energy=np.logspace(-1,5,1000)*u.GeV,
                                            includeIC=includeIC)
        
        #----- Create the model
        model_tot = gammalib.GModels()

        if self.cluster.X_cr_E['X'] > 0: # No need to include the cluster if it is 0
            build_ctools_model.cluster(model_tot,
                                       self.output_dir+'/'+prefix+'_Map.fits',
                                       self.output_dir+'/'+prefix+'_Spectrum.txt',
                                       ClusterName=self.cluster.name)
            
        build_ctools_model.compact_sources(model_tot, self.compact_source)
        
        if obsID is None:
            background = self.obs_setup.bkg
        else:
            background = self.obs_setup.select_obs(obsID).bkg
        build_ctools_model.background(model_tot, background)
        
        model_tot.save(self.output_dir+'/'+prefix+'.xml')
        
