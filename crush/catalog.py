import copy
import json
import os
import zlib

import astropy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from pixell import utils
from sandbox import misc as sbmisc
from scipy import spatial

from . import misc


class HierarchicalCatalog(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        sbmisc.create_dir(self.output_dir)
        self.structure_file = self.output_path('structure.json')
        self.structure = None
        self.higharch_map = {0: None, 1: {}}
        self.catalogs = {0: None, 1: {}, 2: {}}

    def output_path(self, file_name):
        return os.path.join(self.output_dir, file_name)

    def load(self, greedy=False):
        structure = self.__base_load(False)
        if greedy:
            # preload catalogs
            self.get_catalogs_per_freq()
        return structure

    def __base_load(self, reload=False):
        if self.structure is not None and not reload:
            pass
        else:
            try:
                self.structure = json.load(open(self.structure_file, 'r'))
                for freq in self.structure.keys():
                    target_file = self.output_path('{}_hcmap.csv'.format(freq))
                    misc.insert_to_dict(self.higharch_map, read_crush(target_file), [1, freq])
                    for season in self.structure[freq].keys():
                        target_file = self.output_path('{}_{}.csv'.format(freq, season))
                        misc.insert_to_dict(self.catalogs, read_crush(target_file), [2, freq, season])
            except:
                print("failed to load")
        return self.structure

    def get_catalogs_per_season(self):
        self.__base_load(reload=False)
        return self.catalogs[2]

    def get_catalogs_per_freq(self):
        self.__base_load(reload=False)
        if not self.catalogs[1]:
            for freq in self.structure.keys():
                target_file = self.output_path('{}.csv'.format(freq))
                if os.path.exists(target_file):
                    misc.insert_to_dict(self.catalogs, read_crush(target_file), [1, freq])
                else:
                    merged, _ = crossmatch(astropy.table.vstack(list(self.catalogs[2][freq].values())),
                                           0.5 * utils.arcmin)
                    misc.insert_to_dict(self.catalogs, merged, [1, freq])
                    astropy.io.ascii.write(merged, self.output_path('{}.csv'.format(freq)))
        else:
            pass
        return self.catalogs[1]

    def build(self, catalogs, seperation, cuts=[('snr', '>5')], verbose=True, overwrite=False):
        common_cols = None
        structure = {}
        for freq in catalogs.keys():
            structure[freq] = {}
            for season in catalogs[freq].keys():
                structure[freq][season] = ''
                target_file = self.output_path('{}_{}.csv'.format(freq, season))
                if os.path.exists(target_file) and not overwrite:
                    cat = read_crush(target_file)
                    catalogs[freq][season] = cat
                else:
                    if common_cols is None:
                        common_cols = get_common_colnames(misc.nested_dict_values(catalogs))
                    for patch in catalogs[freq][season].keys():
                        cat = catalogs[freq][season][patch]
                        cat = cat[list(common_cols)]
                        for cut in cuts:
                            cat = apply_cut(cat, cut[0], cut[1])
                        catalogs[freq][season][patch] = cat

                    catalogs[freq][season] = _postprocess_catalog(
                        astropy.table.vstack(list(catalogs[freq][season].values())))
                    if verbose: print("crossmatching {} {}".format(freq, season))
                    cat, childs = crossmatch(catalogs[freq][season], seperation, verbose=verbose)
                    for i, child in enumerate(childs):
                        nchild = len(child)
                        if nchild > 1:
                            seasons = cat['season'][i].split(',')
                            patches = cat['patch'][i].split(',')
                            freqs = cat['freq'][i].split(',')

                            note = []
                            for j in range(nchild):
                                note.append('_'.join([seasons[j], patches[j], freqs[j], str(child[j])]))
                            note = ','.join(note)
                            note = str(zlib.compress(note.encode('utf8')))
                            cat[i]['note'] = note
                        cat[i]['child_count'] = int(nchild)
                        cat[i]['season'] = season
                    cat.remove_columns(['patch', 'freq'])
                    idx_col = np.arange(len(cat)).astype(int).astype(object)
                    idx_col = astropy.table.Column(idx_col, name='index')
                    cat.add_column(idx_col)
                    catalogs[freq][season] = cat
                    astropy.io.ascii.write(catalogs[freq][season], self.output_path('{}_{}.csv'.format(freq, season)))
                misc.insert_to_dict(self.catalogs, cat, [2, freq, season])
            target_file = self.output_path('{}_hcmap.csv'.format(freq))

            if os.path.exists(target_file) and not overwrite:
                misc.insert_to_dict(self.higharch_map, read_crush(target_file), [1, freq])
            else:
                cat = astropy.table.vstack(list(catalogs[freq].values()))
                _, qbt, _ = build_kd_tree(cat, seperation)
                t = catalogs[freq][season]['parent', 'child', 'child_count'][:0].copy()
                for i, childs in enumerate(qbt):
                    t.add_row(np.empty(1, dtype=t.dtype)[0])
                    child_dict = {}
                    for child in childs:
                        season = cat[child]['season']
                        idx = cat[child]['index']
                        catalogs[freq][season][idx]['parent'] = i
                        misc.insert_to_dict(child_dict, '', [season, idx])
                    t[i]['child'] = str(child_dict)
                    t[i]['child_count'] = len(childs)

                freq_col = np.array([freq] * len(t)).astype(object)
                freq_col = astropy.table.Column(freq_col, name='freq')
                idx_col = np.arange(len(t)).astype(int).astype(object)
                idx_col = astropy.table.Column(idx_col, name='index')
                t.add_columns([idx_col, freq_col])

                astropy.io.ascii.write(t, self.output_path('{}_hcmap.csv'.format(freq)))
                misc.insert_to_dict(self.higharch_map, t, [1, freq])
                for season in catalogs[freq].keys():
                    astropy.io.ascii.write(catalogs[freq][season], self.output_path('{}_{}.csv'.format(freq, season)))
                    misc.insert_to_dict(self.catalogs, catalogs[freq][season], [2, freq, season])

        if os.path.exists(self.structure_file) and not overwrite:
            structure = json.load(open(self.structure_file, 'r'))
        else:
            json.dump(structure, open(self.structure_file, 'w'))
        self.structure = structure
        return structure


def crossmatch(catalog, seperation, method='kdtree', verbose=True):
    algs = {'kdtree': __kd_tree_matching, 'linear': __linear_matching}
    alg = algs[method]
    crossmatched = catalog[:0].copy()
    return alg(catalog, seperation, crossmatched, verbose)


def build_kd_tree(catalog, seperation):
    coords = np.zeros((len(catalog), 2), dtype=np.float64)
    coords[:, 0] = catalog['ra']
    coords[:, 1] = catalog['dec']
    tree = spatial.cKDTree(coords)
    qbt = tree.query_ball_tree(tree, seperation)
    return tree, qbt, coords


def __kd_tree_matching(catalog, seperation, crossmatched, verbose=True):
    _, qbt, coords = build_kd_tree(catalog, seperation)
    flags = np.zeros(coords.shape[0]).astype(bool)

    elmnt = len(flags)
    delta = int(elmnt / 25)
    delta = min(elmnt, delta)
    ctr = 0
    child = []
    for indxes in qbt:
        if flags[indxes[0]]: continue
        child.append(indxes.copy())
        ctr = np.sum(flags)
        if ctr % delta == 0 and verbose: misc.progress(ctr, elmnt, 'crossmatching')

        crossmatched = coadd_catalog(catalog[indxes], crossmatched)
        for idx in indxes:
            ctr += 1
            flags[idx] = True
    return crossmatched, child


def __linear_matching(catalog, seperation, crossmatched, verbose=True):
    assert False
    ref = catalog.copy()
    coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], unit='radian')

    elmnt = len(coords)
    delta = int(elmnt / 25)
    delta = min(elmnt, delta)
    ctr = 0
    while len(coords) > 0:
        if ctr % delta == 0 and verbose: misc.progress(ctr, elmnt, 'crossmatching')

        coord = coords[0]
        loc = coord.separation(coords) < seperation * astropy.units.rad
        crossmatched = coadd_catalog(ref[loc], crossmatched)
        ref = ref[np.invert(loc)]
        coords = coords[np.invert(loc)]
        ctr += 1
    del ref, coords

    return crossmatched, None


def is_column(catalog, name):
    return name in catalog.colnames


def coadd_catalog(catalog, out_catalog=None):
    ivar = 1. / catalog['sigma_jy'] ** 2
    tot_weight = np.sum(ivar)

    def ivar_sum(vals, ivar=ivar, tot_weight=tot_weight):
        return np.sum((vals * ivar) / tot_weight)

    def char_sum(vals, compress=False):
        ret = ','.join(vals.astype(str).astype(object))
        if compress:
            ret = 'zlib_' + str(zlib.compress(ret.encode('utf8')))
        return ret

    t = catalog[:0].copy()
    t.add_row(np.empty(1, dtype=t.dtype)[0])
    t[0]['ra'] = ivar_sum(catalog['ra'])
    t[0]['dec'] = ivar_sum(catalog['dec'])
    t[0]['jy'] = ivar_sum(catalog['jy'])
    t[0]['sigma_jy'] = 1 / np.sqrt(tot_weight)
    t[0]['snr'] = np.sqrt(np.sum(catalog['snr'] ** 2))

    map_details = ['season', 'patch', 'freq']
    for idx in map_details:
        if is_column(t, idx): t[0][idx] = char_sum(catalog[idx])

    if out_catalog is None:
        out_catalog = t
    else:
        out_catalog = astropy.table.vstack([out_catalog, t])
    return out_catalog


def calibrate_flux(catalog, temp_cal=1.0, pol_cal=1.0, inplace=False):
    ret = catalog if inplace else catalog.copy()
    ret['jy'] *= temp_cal
    ret['sigma_jy'] *= temp_cal
    return ret


def get_common_colnames(catalogs):
    common_cols = None
    for cat in catalogs:
        cur_cols = set(cat.colnames)
        if common_cols is None:
            common_cols = cur_cols
        else:
            common_cols = common_cols & cur_cols
    return common_cols


def standarize_catalogs(catalogs, inplace=False):
    ret = copy.deepcopy(catalogs) if not inplace else catalogs
    common_cols = get_common_colnames(ret)
    for i, cat in enumerate(ret):
        ret[i] = cat[list(common_cols)]
    return ret


def apply_cut(cat, col, constrain, inplace=False):
    exc = "cat['%s']%s" % (col, constrain)
    ret = cat.copy() if not inplace else cat
    try:
        loc = np.where(eval(exc))
        ret = cat[loc]
    except:
        print("can't execute", exc)
    return ret


def _postprocess_catalog(catalog, dtype='astropy'):
    def _convert_table(catalog, dtype='astropy'):
        if dtype not in ['astropy', 'pandas', 'numpy']:
            print("can't recognize dtype={}. return the catalog as astropy table".format(dtype))

        if dtype == 'pandas':
            catalog = catalog.to_pandas()
        elif dtype == 'numpy':
            catalog = np.array(catalog.as_array(), dtype=catalog.dtype).view(np.recarray)
        else:
            pass

        return catalog

    def _add_units(catalog):
        if is_column(catalog, 'ra'): catalog['ra'].unit = u.rad
        if is_column(catalog, 'dec'): catalog['dec'].unit = u.rad
        if is_column(catalog, 'jy'): catalog['jy'].unit = u.Jy
        if is_column(catalog, 'sigma_jy'): catalog['sigma_jy'].unit = u.Jy

        return catalog

    for idx in catalog.colnames:
        if '<U' in str(catalog[idx].dtype):
            catalog[idx] = catalog[idx].astype('object')

    if 'name' in catalog.colnames: catalog['name'] = catalog['name'].astype(object)
    if 'child' in catalog.colnames: catalog['child'] = catalog['child'].astype(str).astype(object)
    if 'parent' in catalog.colnames: catalog['parent'] = catalog['parent'].astype(str).astype(object)
    if 'note' in catalog.colnames: catalog['note'] = catalog['note'].astype(str).astype(object)
    if 'idx' in catalog.colnames: catalog['idx'] = catalog['idx'].astype(str).astype(object)
    if dtype in ['pandas', 'numpy']:
        catalog = _convert_table(catalog, dtype=dtype)
    else:
        catalog = _add_units(catalog)
    return catalog


def read_crush(fname, dtype="astropy"):
    ret = ascii.read(fname)
    return _postprocess_catalog(ret, dtype=dtype)


def read_nemo(fname, dtype='astropy'):
    # # took it from pixell.pointsrcs and modified to be compatible with the output from the latest version of nemo (
    # Oct 14th, 2019)
    """Reads the nemo catalog 
    Args:
        dtype:
        fname:
    """
    ret = ascii.read(fname)
    ret.rename_columns(['RADeg', 'decDeg', 'deltaT_c', 'err_deltaT_c', 'SNR', 'fluxJy', 'err_fluxJy'],
                       ['ra', 'dec', 'I', 'dI', 'snr', 'jy', 'sigma_jy'])
    ret['ra'] *= utils.degree
    ret['dec'] *= utils.degree
    del ret['template'], ret['tileName'], ret['numSigPix'], ret['galacticLatDeg'], ret['name'], ret['I'], ret['dI']
    return _postprocess_catalog(ret)
