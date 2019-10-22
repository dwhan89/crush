import numpy as np
from pixell import enmap, utils
#from soapack import interfaces as sint
from itertools import product
from crush import misc
import scipy.ndimage

def get_bounded_pix(imap, threadhold=None, threshold_factor=1., sigma=None, downsample=None, verbose=False):
    shape, wcs = imap.shape, imap.wcs
    template = imap.copy()

    if not sigma:
        # smooth the input if necessary
        template = scipy.ndimage.gaussian_filter(template, sigma=sigma)
        template = enmap.enmap(template, wcs=wcs)
    if not downsample:
        # downsample maps to speed up the search process
        template = enmap.downgrade(template, factor=downsample)

    loc = np.where(template !=0.)
    if not threadhold:
        threadhold = template.mean()
    loc = np.where(template>threadhold*threshold_factor)

    binary_map = template*0.
    binary_map[loc] = 1.
    del template

    # get pix for the bounded region
    _, pix = misc.max_size(binary_map, value=1., varbose=verbose)
    # if the input map was downsampled, we need to map pixels at low resolutions to those at higher resolution
    if not downsample:
        ll_pix = imap.sky2pix(binary_map.pix2sky([pix[0], pix[2]]))
        ur_pix = imap.sky2pix(binary_map.pix2sky([pix[1], pix[3]]))
        pix = np.array([ll_pix[0], ur_pix[0], ll_pix[1], ur_pix[1]]).astype(int)

    return pix

