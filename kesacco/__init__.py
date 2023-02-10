__version__ = '1.1.4'

try:
    from kesacco import clustpipe
except ImportError:
    print('WARNING: Could not import clusterpipe from kesacco.')
    print('         You may try (re)installing dependencies by')
    print('         hand. For example running:                ')
    print('             $ conda install matplotlib            ')
    print('             $ conda install numpy                 ')
    print('             $ conda install scipy                 ')
    print('             $ conda install astropy               ')
    print('             $ conda install ctools                ')
    print('             $ conda install emcee                 ')
    print('             $ conda install corner                ')
    print('             $ pip install minot                   ')
