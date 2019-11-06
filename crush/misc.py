import os
from collections import namedtuple
from functools import reduce
from operator import mul
import numpy as np
from sandbox import misc as sbmisc


def insert_to_dict(dictionary, elmt, indexes):
    depth = len(indexes)
    key = indexes[0]
    if depth > 1:
        indexes = indexes[1:]
        if key not in dictionary.keys(): 
            dictionary[key] = {}
        insert_to_dict(dictionary[key], elmt, indexes)
    else:
        dictionary[key] = elmt


def has_key(nested_dict, keys):
    ''' search through nested dictionary to fine the elements '''
    if not type(keys) is tuple: keys = (keys,)
    if not type(nested_dict) == dict: return False
    if(len(keys) > 1):
        has_it = keys[0] in nested_dict
        return has_key(nested_dict[keys[0]], keys[1:]) if has_it else False
    else:
        return keys[0] in nested_dict


def get_from_dict(nested_dict, keys, safe=True):
    if not type(keys) is tuple: keys = (keys,)
    if safe and not has_key(nested_dict, keys): return None

    if(len(keys) > 1):
        return get_from_dict(nested_dict[keys[0]], keys[1:], False)
    else:
        return nested_dict[keys[0]]


def NestedDictValues(dictionary):
    for v in dictionary.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v

def progress(count, total, status=''):
    # adoped from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 60
    percents = 0
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))

def join_char_array(arrays, seperator='_'):
    sep_array = np.array([seperator]*len(arrays[0]))

    ret = arrays[0]
    for array in arrays[1:]:
        ret = np.core.defchararray.add(ret,sep_array)
        ret = np.core.defchararray.add(ret,array)
    return ret

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
