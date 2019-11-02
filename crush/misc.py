import os
from collections import namedtuple
from functools import reduce
from operator import mul

import numpy as np


##
# some io stuff

def create_dir(path2dir, safe=True):
    if safe: assert (not os.path.exists(path2dir))
    os.mkdir(path2dir)
    return 0


##
# Find the largest bound area. Based on the solution on https://stackoverflow.com/questions/2478447

RectangleGeo = namedtuple('RectangleGeo', 'start height')


def max_size(mat, value=1, verbose=False):
    """Find height, width, and idxes of the largest rectangle constraining all
    Based on the solution on https://stackoverflow.com/questions/247844
    """
    it = iter(mat)
    hist = [(el == value) for el in next(it, [])]

    def _box_idxes(yidxes, xidxes):
        return yidxes + xidxes

    max_size, xidxes = max_rectangle_size(hist)
    max_idxes = _box_idxes((0, 0), xidxes)
    nrow = mat.shape[0]
    nstep = int(nrow / 50)
    for i, row in enumerate(it, start=1):
        if verbose and i % nstep == 0:
            print("[%d/%d] processed" % (i, nrow))
        hist = [(1 + h) if el == value else 0 for h, el in zip(hist, row)]
        cur_size, xidxes = max_rectangle_size(hist)
        cur_size = max(max_size, cur_size, key=area)
        if max_size != cur_size:
            max_size, max_idxes = (cur_size, _box_idxes((i - hist[xidxes[0]] + 1, i), xidxes))

    return np.array(max_size).astype(int), max_idxes


def max_rectangle_size(histogram):
    """Find height, width, idxes of the largest rectangle that fits entirely under
    the histogram. Based on the solution on https://stackoverflow.com/questions/247844
    """
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0)  # height, width of the largest rectangle
    pos = 0  # current position in the histogram
    xidxes = (0, 0)  # xidxes of the rectangle

    def _update(max_size, xidxes, height, start, end):
        cur_size = max(max_size, (height, (end - start)), key=area)
        if max_size != cur_size:
            max_size, xidxes = cur_size, (start, end - 1)
        return max_size, xidxes

    for pos, height in enumerate(histogram):
        start = pos  # position where rectangle starts

        while True:
            if not stack or height > top().height:
                stack.append(RectangleGeo(start, height))  # push
            elif stack and height < top().height:
                max_size, xidxes = _update(max_size, xidxes, top().height, top().start, pos)
                start, _ = stack.pop()
                continue
            break  # height == top().height goes here
    # processing the last element in the stack
    pos += 1
    for start, height in stack:
        max_size, xidxes = _update(max_size, xidxes, height, start, pos)
    return max_size, xidxes


def area(size):
    return reduce(mul, size)
