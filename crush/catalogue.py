import numpy as np
from astropy.io import ascii

def read_nemo(fname):
    ## took it from pixell.pointsrcs and modified to be compatible with the output from the latest version of nemo (Oct 14th, 2019)
    """Reads the nemo ascii catalog format, and returns it as a recarray."""
    tbl = ascii.read(fname)
    tbl.rename_columns(['RADeg', 'decDeg', 'deltaT_c', 'err_deltaT_c'], ['ra', 'dec', 'I', 'dI'] )
    ocat = np.array(tbl.as_array(), dtype=tbl.dtype).view(np.recarray)
    return ocat

