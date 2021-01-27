# KESACCO: Keen Event Simulation and Analysis for CTA Cluster Observations

This repository contains a software dedicated to the simulation of CTA event files 
corresponding to the observation of galaxy clusters. The event files can then be analyzed 
to produce CTA analysis results.
                                                            
- clustpipe.py : 
	Main code that defines the class ClusterPipe.
    
- clustpipe_common.py : 
  Subclass that defines common tools.
   
- clustpipe_sim.py : 
  Subclass that deal with event simulation.
        
- clustpipe_ana.py : 
  Subclass that deal with observation analysis.
    
- clustpipe_ana_plot.py : 
  Sub-modules dedicated to run automatic plots related to analysis.

- clustpipe_title.py : 
	Title for the software.

- Tools :
  Repository that gather several useful libraries and run ctools scripts.
  It also contain the compact source, background, and observation setup 
  classes.

- notebook :
	Repository where to find Jupyter notebook used for development/example. 

## Overview of the physical processes and structure of the code
<figure>
	<img src="/overview.png" width="600" />
	<figcaption> Figure 1. Overview of the KESACCO software.</figcaption>
</figure>

<p style="margin-bottom:3cm;"> </p>

## Environment
The code works with python 3. It should work with python 2, although several features may fail. Please make sure that you are in the correct environment when you run the code. In addition, the kesacco directory should be in your python path so it can be found.

## Installation
To install these tools, just fork the repository to your favorite location in your machine.
The software depends on standard python package (non-exhaustive list yet):
- astropy
- matplotlib
- random
- numpy
- os
- copy
- pickle

But also:
- gammalib (http://cta.irap.omp.eu/gammalib/)
- ctools (http://cta.irap.omp.eu/ctools/)
- minot (see https://github.com/remi-adam/minot)
