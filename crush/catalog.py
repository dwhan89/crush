import numpy as np
import numpy as np
from astropy import units as u
from astropy.io import ascii
from pixell import utils


def apply_cut(cat, col, constrain, inplace=False):
    exc = "cat['%s']%s" % (col, constrain)
    ret = cat.copy() if not inplace else cat
    try:
        loc = np.where(eval(exc))
        ret = cat[loc]
    except:
        print("can't execute", exc)
    return ret


def read_nemo(fname, dtype='astropy'):
    ## took it from pixell.pointsrcs and modified to be compatible with the output from the latest version of nemo (Oct 14th, 2019)
    """Reads the nemo catalog 
    Args:
        fname:
    """
    if dtype not in ['astropy', 'pandas', 'numpy']:
        print("can't recognize dtype={}. return the catalog as astropy table".format(dtype))

    ret = ascii.read(fname)
    ret.rename_columns(['RADeg', 'decDeg', 'deltaT_c', 'err_deltaT_c', 'SNR', 'fluxJy', 'err_fluxJy'],
                       ['ra', 'dec', 'I', 'dI', 'snr', 'jy', 'sigma_jy'])
    ret['ra'] *= utils.degree
    ret['dec'] *= utils.degree

    if dtype == 'pandas':
        ret = ret.to_pandas()
    elif dtype == 'numpy':
        ret = np.array(ret.as_array(), dtype=ret.dtype).view(np.recarray)
    else:
        ret['ra'] *= u.rad
        ret['dec'] *= u.rad
        ret['jy'] *= u.Jy
        ret['sigma_jy'] *= u.Jy

    return ret
