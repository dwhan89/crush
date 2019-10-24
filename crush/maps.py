import numpy as np
from pixell import enmap, utils
# from soapack import interfaces as sint
from itertools import product
from crush import misc
import scipy.ndimage


def nrect_grid(imap, grid_extent):
    return np.ceil(imap.extent() / grid_extent).astype(np.int)


def rect_grid_edges(shape, ngrids):
    ny, my = divmod(shape[-2], ngrids[0])
    nx, mx = divmod(shape[-1], ngrids[1])

    ret = np.zeros((ngrids[0], ngrids[1], 4))
    for j, nygrid in enumerate(range(ngrids[0])):
        for i, nxgrid in enumerate(range(ngrids[1])):
            xsidx = nx * i + min(i, mx)
            ysidx = ny * j + min(j, my)

            xeidx = xsidx + nx
            yeidx = ysidx + ny

            if i >= mx: xeidx -= 1
            if j >= my: yeidx -= 1

            xeidx = min(xeidx, shape[-1] + 1)
            yeidx = min(yeidx, shape[-2] + 1)

            ret[j, i, :] = np.array([ysidx, yeidx, xsidx, xeidx])
i
    return ret.astype(np.int)


def threshold_grids(imap, grid_edges, eps=0.):
    ny, nx = grid_edges.shape[0], grid_edges.shape[1]
    ret = np.ones((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            ys, ye, xs, xe = grid_edges[j, i]
            loc = np.where(imap[ys:ye + 1, xs:xe + 1] > eps)
            if len(loc[0]) == 0:
                ret[j, i] = False
    return ret


def bounded_pix(imap, threadhold=None, threshold_factor=1., sigma=None, downsample=None, verbose=False):
    shape, wcs = imap.shape, imap.wcs
    template = imap.copy()

    if sigma:
        # smooth the input if necessary
        template = scipy.ndimage.gaussian_filter(template, sigma=sigma)
        template = enmap.enmap(template, wcs=wcs)
    if downsample:
        # downsample maps to speed up the search process
        if verbose: "downsampling by a factor of %d" % downsample
        template = enmap.downgrade(template, factor=downsample)

    loc = np.where(template != 0.)
    if not threadhold:
        threadhold = template.mean()
    loc = np.where(template > threadhold * threshold_factor)

    binary_map = template * 0.
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


def bounded_pixs(imap, grid_pix=None, valid_grid=None, threshold=None, threshold_factor=1., sigma=None,
                 downsample=None, verbose=False):
    if grid_pix is None:
        grid_pix = np.zeros((1, 1, 4))
        grid_pix[0, 0][:] = np.array([0, imap.shape[0] - 1, 0, imap.shape[1] - 1])
    grid_pix = grid_pix.astype(np.int)

    nxgrid, nygrid = grid_pix.shape[:2]
    if valid_grid is None:
        valid_grid = np.ones((nygrid, nxgrid)).astype(bool)

    ret = np.zeros(grid_pix.shape).astype(np.int)
    for j in range(nxgrid):
        for i in range(nygrid):
            if not valid_grid[j, i]:
                # not valid grid. skip it
                continue
            else:
                ys, ye, xs, xe = grid_pix[j, i]
                subsect_pix = bounded_pix(imap[ys:ye + 1, xs:xe + 1], threshold, threshold_factor, sigma, downsample,
                                          verbose)
                ret[j, i, :] = subsect_pix
                ret[j, i, :2] += ys
                ret[j, i, 2:] += xs
    return ret


def gridpix2sky(shape, wcs, grid_pixs):
    ret = np.zeros(grid_pixs.shape)
    for j in range(grid_pixs.shape[0]):
        for i in range(grid_pixs.shape[1]):
            pix = grid_pixs[j,i]
            ll_coords = enmap.pix2sky([pix[0], pix[2]])
            ur_coords = enmap.pix2sky([pix[1], pix[3]])
            ret[j,i,:] = np.array([ll_coords[0], ur_coords[0], ll_coords[1], ur_coords[1]])
    return ret






