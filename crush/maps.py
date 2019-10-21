from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

##
# Find the largest bound area. Based on the solution on https://stackoverflow.com/questions/2478447


Info = namedtuple('Info', 'start height')


def max_size(mat, value=0):
    """Find height, width, and idxes of the largest rectangle constraining all
    Based on the solution on https://stackoverflow.com/questions/247844
    """
    it = iter(mat)
    hist = [(el == value) for el in next(it, [])]
    max_size, max_idxes = max_rectangle_size(hist)
    for row in it:
        hist = [(1 + h) if el == value else 0 for h, el in zip(hist, row)]
        cur_size, cur_idxes = max_rectangle_size(hist)
        cur_size = max(max_size, cur_size, key=area)
        #print(cur_idxes)
        if  max_size != cur_size:
            max_size, max_idxes = (cur_size, cur_idxes)

    return np.array(max_size).astype(int), max_idxes


def max_rectangle_size(histogram):
    """Find height, width, idxes of the largest rectangle that fits entirely under
    the histogram. Based on the solution on https://stackoverflow.com/questions/247844
    """
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0)  # height, width of the largest rectangle
    pos = 0  # current position in the histogram
    xidxes = (0,0) # xidxes of the rectangle

    def _update(max_size, xidxes, height, start, end):
        cur_size = max(max_size, xidxes, (height, (end - start)), key=area)
        if max_size != cur_size:
            max_size, xidxes = cur_size, (start, end - 1)
        return max_size, xidxes

    for pos, height in enumerate(histogram):
        start = pos  # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height))  # push
            elif stack and height < top().height:
                max_size, xidxes = _update(max_size, xidxes, top().height, top().start, pos)
                start, _ = stack.pop()
                continue
            break  # height == top().height goes here

    # processing the last element in the stack
    pos += 1
    for start, height in stack:
        max_size, idxes = _update(max_size, xidxes, height, start, pos)
    return max_size, xidxes

def area(size):
    return reduce(mul, size)
