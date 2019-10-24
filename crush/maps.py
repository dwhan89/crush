import numpy as np
from pixell import enmap, utils
#from soapack import interfaces as sint
from itertools import product
from crush import misc
import scipy.ndimage

def nrect_grid(imap, grid_extent):
    return np.ceil(imap.extent()/grid_extent).astype(np.int)

def rect_grid_edges(shape, ngrids):
    ny, my = divmod(shape[-2], ngrids[0])
    nx, mx = divmod(shape[-1], ngrids[1])

    ret = np.zeros((ngrids[0],ngrids[1],4)) 
    for j, nygrid in enumerate(range(ngrids[0])):
        for i, nxgrid in enumerate(range(ngrids[1])):
            xsidx = nx*i + min(i,mx)
            ysidx = ny*j + min(j,my)

            xeidx = xsidx + nx
            yeidx = ysidx + ny

            if i >= mx: xeidx -= 1
            if j >= my: yeidx -= 1

            xeidx = min(xeidx, shape[-1]+1)
            yeidx = min(yeidx, shape[-2]+1)

            ret[j, i, :] = np.array([ysidx, yeidx, xsidx, xeidx]) 

    return ret.astype(np.int)

def bounded_pix(imap, threadhold=None, threshold_factor=1., sigma=None, downsample=None, verbose=False):
    shape, wcs = imap.shape, imap.wcs
    template = imap.copy()

    if sigma:
        # smooth the input if necessary
        template = scipy.ndimage.gaussian_filter(template, sigma=sigma)
        template = enmap.enmap(template, wcs=wcs)
    if downsample:
        # downsample maps to speed up the search process
        if verbose: "downsampling by a factor of %d" %downsample
        template = enmap.downgrade(template, factor=downsample)

    loc = np.where(template !=0.)
    if not threadhold:
        threadhold = template.mean()
    loc = np.where(template>threadhold*threshold_factor)

    binary_map = template*0.
    binary_map[loc] = 1.
    del template

    # get pix for the bounded region
    _, pix = misc.max_size(binary_map, value=1., verbose=verbose)
    # if the input map was downsampled, we need to map pixels at low resolutions to those at higher resolution
    if downsample:
        ll_pix = imap.sky2pix(binary_map.pix2sky([pix[0], pix[2]]))
        ur_pix = imap.sky2pix(binary_map.pix2sky([pix[1], pix[3]]))
        pix = np.array([ll_pix[0], ur_pix[0], ll_pix[1], ur_pix[1]]).astype(int)

    return pix