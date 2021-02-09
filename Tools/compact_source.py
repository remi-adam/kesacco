"""
This file contains the CompactSource class. It is dedicated to the construction of a 
CompactSource object, which can be used in the sky model for CTA simulations. It is 
dedicated to include mainly point source AGN in clusters, but could eventually include 
resolved AGN.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u


#==================================================
# Cluster class
#==================================================

class CompactSource(object):
    """ PointSource class. 
    This class defines a PointSource object. This is a list of
    individual point source, with coresponding parameters.
        
    Attributes
    ----------  
    - name     : the name of the sources
    - spatial  : the spatial properties of the sources
    - spectral : the spectral properties of the sources
    - temporal : the temporal properties of the sources
    - redshift : the redshift of the sources (for EBL)

    Methods
    ----------  
    - add_source
    - remove_source
    - print_source

    """
    
    #==================================================
    # Initialize the CompactSource object
    #==================================================

    def __init__(self):
        """
        Initialize the PointSource object.
        By default the list is empty.
        
        Parameters
        ----------
        - name (str): the label of the source
        - spatial (dict) : the spatial properties of the source
        - spectral (dict) : the spectral properties of the source
        - temporal (dict) : the temporal properties of the source
        - redshift (float) : the redshift of the source (for EBL)

        """
        
        self.name     = []
        self.spatial  = []
        self.spectral = []
        self.temporal = []
        self.redshift = []
        
        
    #==================================================
    # Add a new source
    #==================================================

    def add_source(self,
                   name='Source',
                   spatial={'type':'PointSource',
                            'param':{'RA':{'value':0.0*u.deg, 'free':False},
                                     'DEC':{'value':0.0*u.deg, 'free':False}}},
                   spectral={'type':'PowerLaw',
                             'param':{'Prefactor':{'value':1e-7*u.Unit('m-2 s-1 TeV-1'), 'free':True},
                                      'Index':{'value':2.5, 'free':True},
                                      'PivotEnergy':{'value':1.0*u.TeV, 'free':False}}},
                   temporal={'type':'Constant',
                             'param':{'Normalization':{'value':1.0, 'free':False}}},
                   redshift=None
    ):

        
        """
        Add a new source to the list. The definition of the parameters 
        follows that used by ctools:
        http://cta.irap.omp.eu/ctools/users/user_manual/models_implementation.html
        http://cta.irap.omp.eu/ctools/users/user_manual/models.html
        
        In case a parameter list is given in the dictionary, this can include:
        'name', 'scale', 'value', 'min', 'max', 'error', 'free'
        
        Parameters
        ----------
        - name (str): name used for source label
        - spatial (dictionary): contain coordinates of the source
        - spectral (dictionary): spectral properties of the source
        - temporal (dictionary): time properties of the source
        - redshift (float): give the redshift to be used for EBL
        
        """

        w = np.where(np.array(self.name).astype('str') == name)[0]

        if len(w) > 0:
            print("!!!! WARNING !!!! The source you are trying to add already exist.")
            print("                  Doing nothing.")
        else:
            self.name.append(name)
            self.spatial.append(spatial)
            self.spectral.append(spectral)
            self.temporal.append(temporal)
            self.redshift.append(redshift)
        
        
    #==================================================
    # Remove a source
    #==================================================

    def delete_source(self, name):
        """
        Delete a source from the list.
        
        Parameters
        ----------
        - name (str): name used for source label
        
        """

        w = np.where(np.array(self.name).astype('str') == name)[0]

        if len(w) > 1:
            raise ValueError("Unexpected error: the given name is matched to several existing source.")

        if len(w) == 0:
            raise ValueError("The given source does not exist.")

        if len(w) == 1:
            idx = w[0]
            del self.name[idx]
            del self.spatial[idx]
            del self.spectral[idx]
            del self.temporal[idx]
            del self.redshift[idx]
        
    #==================================================
    # Remove a source
    #==================================================

    def print_source(self):
        """
        Show the source list in a fency way.
        
        Parameters
        ----------
        
        """

        Ncomp = len(self.name)

        if Ncomp == 0:
            print('=== No source is currently defined')
        
        for k in range(Ncomp):

            #----- First show the name
            print('--- '+self.name[k]+' at z='+str(self.redshift[k]))

            #----- Show the spatial component
            print('    -- Spatial model: '+self.spatial[k]['type'])

            if 'param' in self.spatial[k].keys():
                for key in self.spatial[k]['param'].keys():
                    print('         '+key+': '+str(self.spatial[k]['param'][key]))

            #----- Show the spectral component
            print('    -- Spectral model: '+self.spectral[k]['type'])

            if 'param' in self.spectral[k].keys():
                for key in self.spectral[k]['param'].keys():
                    print('         '+key+': '+str(self.spectral[k]['param'][key]))

            #----- Show the temporal component
            print('    -- Temporal model: '+self.temporal[k]['type'])

            if 'param' in self.temporal[k].keys():
                for key in self.temporal[k]['param'].keys():
                    print('         '+key+': '+str(self.temporal[k]['param'][key]))
