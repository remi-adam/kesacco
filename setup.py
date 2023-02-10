from setuptools import setup, find_packages 
from codecs import open
from os import path
from kesacco.__init__ import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# See https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/ for information about python packaging
setup(
    name='kesacco',
    version=__version__,
    description='Keen Event Simulation and Analysis for CTA Cluster Observations: python code to simulate and analyse galaxy cluster data with CTA',
    long_description=long_description,  #this is the readme 
    long_description_content_type='text/markdown',
    url='https://github.com/remi-adam/kesacco',
    author='Remi Adam',
    author_email='remi.adam@oca.eu',
    license='BSD',
    # See https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
    packages = find_packages(),
    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'astropy >= 1.2.1',
        'matplotlib',
        'ebltable',
        'emcee',
        'corner',
        'ctools',
        'minot'
    ]
)
