"""
This file contains the Common class, which lists 
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
from astropy.table import Table
from astropy.io import fits
import cscripts

from kesacco.Tools import make_cluster_template
from kesacco.Tools import build_ctools_model
from kesacco.Tools import utilities


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
    - config_save(self)
    - config_load(self, config_file)
    - _check_obsID(self, obsID)
    - _correct_eventfile_names(self, xmlfile, prefix='Events')
    - _write_new_xmlevent_from_obsid(self, xmlin, xmlout, obsID)
    - _rm_source_xml(self, xmlin, xmlout, source)
    - _match_cluster_to_pointing(self, extra=1.1)
    - _match_anamap_to_pointing(self, extra=1.1)
    - _make_model(self, prefix='Model', includeIC=False)
    - _define_std_filenames(self)
    - _load_onoff_region(self, filename)

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
        Validate the obsID given by the user (e.g. remove duplicate
        obsid, check that the obsid exists, make sure of the format).
        
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
        by naming them using the obsid.
        
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
    
    def _write_new_xmlevent_from_obsid(self, xmlin, xmlout, obsID, obs_setup,
                                       updateIRF=True):
        """
        Read the xml file gathering the event list, remove the event
        files that are not selected, and write a new xml file.
        
        Parameters
        ----------
        - xmlin (str): input xml file
        - xmlout (str): output xml file
        - obsID (list): list of str
        - obs_setup (dict): the observation setup
        - updateIRF (bool): update the IRF or force the true simulated IRF 
        (this is in case you want to simulate the data with a given IRF, but 
        analyse them with another one)

        Outputs
        -------
        - The new xml file is writen
        """

        xml     = gammalib.GXml(xmlin)
        obslist = xml.element('observation_list')
        obsid_in = []
        
        # Remove unwanted obsid
        for i in range(len(obslist))[::-1]:
            if obslist[i].attribute('id') not in obsID:
                obslist.remove(i)
            else:
                obsid_in.append(obslist[i].attribute('id'))
                
        # Warning if wanted obsid not available
        for i in range(len(obsID)):
            if obsID[i] not in obsid_in:
                print('WARNING: Event file with obsID '+obsID[i]+' does not exist. It is ignored.')

        # Update IRF
        if updateIRF:
            obslist = xml.element('observation_list')
            for i in range(len(obslist)):
                wobsid = np.where(np.array(obs_setup.obsid) == obslist[i].attribute('id'))[0]
                if len(wobsid) != 1:
                    raise ValueError('Problem with obsid matching')
                db   = obs_setup.caldb[wobsid[0]]
                resp = obs_setup.irf[wobsid[0]]
                obslist[i][0].attribute(1).value(db)   # database
                obslist[i][0].attribute(2).value(resp) # response

        # Save
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
        Match the ClusterPipe map according to the pointing list.
        
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
    
    def _make_model(self, prefix='Model', includeIC=False, obsID=None):
        """
        This function is used to construct the overall model (unstacked) 
        and save the xml file.
        
        Parameters
        ----------
        - prefix (str): text to add as a prefix of the file names
        - includeIC (bool): include inverse Compton in the model
        
        """
        
        #----- Make cluster template files
        if (not self.silent) and (self.cluster.map_reso > 0.01*u.deg):
            print('------------------------------------------------------------------')
            print('WARNING: the FITS map resolution (self.cluster.map_reso) is larger')
            print('         than 0.01 deg, while the PSF at 100 TeV is ~0.02 deg.    ')
            print('------------------------------------------------------------------')
            print('')
        
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

        if self.cluster.X_crp_E['X'] > 0: # No need to include the cluster if it is 0
            build_ctools_model.cluster(model_tot,
                                       self.output_dir+'/'+prefix+'_Map.fits',
                                       self.output_dir+'/'+prefix+'_Spectrum.txt',
                                       ClusterName=self.cluster.name,
                                       tscalc=True)
            
        build_ctools_model.compact_sources(model_tot,
                                           self.compact_source,
                                           self.output_dir,
                                           tscalc=True,
                                           EBL_model=self.cluster.EBL_model,
                                           energy=np.logspace(-1,5,1000)*u.GeV)
        
        if obsID is None:
            background = self.obs_setup.bkg
        else:
            background = self.obs_setup.select_obs(obsID).bkg
        build_ctools_model.background(model_tot, background)

        model_tot.save(self.output_dir+'/'+prefix+'_Unstack.xml')

        return model_tot
    

    #==================================================
    # Define standard file name
    #==================================================
    
    def _define_std_filenames(self):
        """
        This function defines the standard filenames.
        
        Parameters
        ----------

        Outputs
        -------
        - inobs (str): input observation file, or
        - inmodel (str): input model
        - expcube (str): 
        - psfcube (str): 
        - bkgcube (str): 
        - edispcube (str): 
        - modcube (str): 
        - modcubeCl (str): 

        """

        #----- Start with general unbinned names
        inobs     = self.output_dir+'/Ana_EventsSelected.xml'
        inmodel   = self.output_dir+'/Ana_Model_Input.xml'
        expcube   = None
        psfcube   = None
        bkgcube   = None
        edispcube = None
        modcube   = self.output_dir+'/Ana_Model_Cube.fits'         # Always the same because accounts
        modcubeCl = self.output_dir+'/Ana_Model_Cube_Cluster.fits' # for likelihood fit model and stack

        if self.method_ana == 'ONOFF':
            if self.method_stack:
                inobs   = self.output_dir+'/Ana_ObsDef_OnOff_Stack.xml'
                inmodel = self.output_dir+'/Ana_Model_Input_OnOff_Stack.xml'
            else:
                inobs   = self.output_dir+'/Ana_ObsDef_OnOff_Unstack.xml'
                inmodel = self.output_dir+'/Ana_Model_Input_OnOff_Unstack.xml'
        else:
            if self.method_binned:
                if self.method_stack:
                    inobs       = self.output_dir+'/Ana_Countscube.fits'
                    inmodel     = self.output_dir+'/Ana_Model_Input_Stack.xml'
                    expcube     = self.output_dir+'/Ana_Expcube.fits'
                    psfcube     = self.output_dir+'/Ana_Psfcube.fits'
                    bkgcube     = self.output_dir+'/Ana_Bkgcube.fits'
                    if self.spec_edisp:
                        edispcube = self.output_dir+'/Ana_Edispcube.fits'
                else:
                    inobs   = self.output_dir+'/Ana_Countscube.xml'
                    #inobs   = self.output_dir+'/Ana_EventsSelected.xml'
                    inmodel = self.output_dir+'/Ana_Model_Input_Unstack.xml'
                    
                # inobs = self.output_dir+'/Ana_ObsDef.xml'
                
        return inobs, inmodel, expcube, psfcube, bkgcube, edispcube, modcube, modcubeCl


    #==================================================
    # Load DS9 ONOFF region
    #==================================================
    
    def _load_onoff_region(self, filename):
        """
        This function loads DS9 ONOFF regions.
        
        Parameters
        ----------
        - filename (str): full path to the DS9 reg file

        Outputs
        -------
        - region (list): list of regions in the file

        """

        reg = []
        f = open(filename, "r")
        Lines = f.readlines() 
        for i in range(2,len(Lines)):
            txtlist = (((Lines[i][11:]).split(')\n'))[:-1])[0].split(',')
            numlist = []
            for index in range(len(txtlist)):
                numlist.append(float(txtlist[index]))
            reg.append(numlist)
        
        return reg

    
    #==================================================
    # define bin file using csebins
    #==================================================
    
    def def_binfile_csebins(self, irf, caldb, outfile,
                            aeffthres=0.2, bkgthres=0.5):
        """
        Generates energy boundaries for stacked analysis.
        (http://cta.irap.omp.eu/ctools/users/reference_manual/csebins.html#csebins)
        
        Parameters
        ----------
        - irf (str): name of instrument response function (e.g. North_z20_50h)
        - caldb (str): name of calibration database (e.g. prod3b-v2)
        - outfile (file): Name of the energy boundary output file.
        - aeffthres (real): Fractional change in effective area that leads to insertion of a new energy boundary.
        - bkgthres (real): Fractional change in background rate that leads to insertion of a new energy boundary. 
    
        Outputs
        --------
        outfile is created and can be used in the analysis
        """
        
        bining = cscripts.csebins()
        bining['inobs']     = 'NONE'
        bining['irf']       = irf
        bining['caldb']     = caldb
        bining['outfile']   = outfile
        bining['emin']      = self.spec_emin.to_value('TeV')
        bining['emax']      = self.spec_emax.to_value('TeV')
        bining['aeffthres'] = aeffthres
        bining['bkgthres']  = bkgthres
        bining.execute()
    
        if not self.silent:
            print('     ---> def_binfile_csebins - number of bins is :', bining.ebounds().size()-1)

            
    #==================================================
    # define bin file using arbitrary values
    #==================================================
    
    def def_binfile_arb(self, bin_edges, outfile):
        """
        Generates energy boundaries for stacked analysis.
        
        Parameters
        ----------
        - bin_edges (qunatity): the value of the bin edges (Nbin + 1 points).
        - outfile (file): Name of the energy boundary output file.
    
        Outputs
        --------
        outfile is created and can be used in the analysis
        """

        # work in keV to mimic csebins
        bin_edges = bin_edges.to_value('keV')

        # Check the the bining is fine with emin, emax
        if self.spec_emin.to_value('keV') > np.amin(bin_edges):
            print('WARNING: the lower bin is lower than parameter emin. Check bin_edges.')
        if self.spec_emax.to_value('keV') < np.amin(bin_edges):
            print('WARNING: the higher bin is higher than parameter emin. Check bin_edges.')

        # Collect Emin,Emax for each bin
        Emin = bin_edges[0:len(bin_edges)-1]
        Emax = bin_edges[1:len(bin_edges)]

        # build the table
        bining = Table([Emin, Emax], names=['E_MIN', 'E_MAX'], units=['keV', 'keV'])

        # build the HDUL and save the file
        primary_hdu = fits.PrimaryHDU()
        bining_hdu = fits.BinTableHDU(bining, name='EBOUNDS')
        hdul = fits.HDUList([primary_hdu, bining_hdu])
        hdul.writeto(outfile, overwrite=True)

        if not self.silent:
            print('     ---> def_binfile_arb - bin edges are :', bin_edges*1e-6, 'GeV')
