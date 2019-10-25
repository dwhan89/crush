from crush import config, maps
from soapack import interfaces as soint
from pixell import enmap, utils
import numpy as np
import os
import yaml

strict = True

nemo_config_file = config.package_data_path('configs/nemo.yaml')
template =  config.package_data_path('configs/nemo_template.yaml')
nemo_config = config.read_yaml(nemo_config_file)

# load data model
DM = soint.models['act_mr3']()

print(nemo_config)
patches = nemo_config['nemo']['patches']
if type(patches) != list:
    assert (patches in soint.models.keys())  # make sure data model is available
    # load all available patches in that model
    patches = DM.get_psa_indexes()
else:
    pass

for patch in patches:
    season, patch, array, freq = patch.split('_')
    arr_freq = '%s_%s' % (array, freq)

    beam_version = nemo_config['act_mr3']['beam_version']
    beam_file = DM.get_beam_fname(season, patch, arr_freq, version=beam_version, realspace=True)
    map_file = DM.get_coadd_fname(season, patch, arr_freq, srcfree=False)
    weight_file = DM.get_coadd_ivar_fname(season, patch, arr_freq)

    if strict:
        file_lists = [beam_file, map_file, weight_file]
        for file_path in file_lists:
            try:
                assert (os.path.exists(file_path))
            except AssertionError:
                print("Missing File: %s" % file_path)
                exit(1)


    # build tile if needed
    tile_setting = nemo_config['nemo']['tiles'].split('&')

    ngrids = [1,1]
    # start automatic tile generation
    noise_tiles = {'autoBorderDeg': 0.5}
    tiles = []
    if 'auto' in tile_setting:
        ivar = enmap.read_fits(weight_file)
        shape, wcs = ivar.shape, ivar.wcs
        default_extent = np.array(nemo_config['nemo']['default_tile_extent'])*utils.degree
        ngrids = maps.nrect_grid(ivar, default_extent)
        nygrid, nxgrid = ngrids

        # some simplification
        if nxgrid*nygrid <= 8:
            ngrids = [1,1]
            nxgrid = nygrid = 1

        grid_pix = maps.rect_grid_pix(shape, ngrids)
        valid_grid = maps.threshold_grids(ivar, grid_pix)

        # put this in the setting!
        threshold_factor = 1.3 if patch not in ['cmb', 'boss'] else 0.8

        noise_grid_pix =  maps.bounded_pixs(ivar, grid_pix, valid_grid, threshold_factor=threshold_factor , sigma=2, downsample=10)

        grid_coords = maps.gridpix2sky(shape, wcs, grid_pix)/utils.degree
        noise_grid_coords = maps.gridpix2sky(shape, wcs, noise_grid_pix)
        noise_grid_coords = maps.reguarlize_rect_grid(noise_grid_coords/utils.degree)
        for j in np.arange(nygrid):
            for i in np.arange(nxgrid):
                if not valid_grid[j,i]: continue
                decs, dece, ras, rae = grid_coords[j,i].tolist()
                tiles.append({'tileName':"{}_{}".format(j,i), 'RADecSection': [ras, rae, decs, dece]})
                decs, dece, ras, rae = noise_grid_coords[j,i].tolist()
                noise_tiles["{}_{}".format(j,i)] = [ras, rae, decs, dece]
    if 'custom' in tile_setting:
        # do something here
        pass

    mpi_switch = (ngrids[0]*ngrids[1] == 1)

    template['unfilteredMaps'][0]['mapFileName'] = map_file
    template['unfilteredMaps'][0]['weightsFileName'] = weight_file
    template['unfilteredMaps'][0]['beamFileName'] = beam_file
    template['unfilteredMaps'][0]['obsFreqGHz'] = 95.0 if freq == 'f090' else 148.0
    template['useMPI'] = mpi_switch
    template['thresholdSigma'] = nemo_config['nemo']['snr']
    template['objIdent'] = 'mr3c_{}-'.format(patch)
    template['catalogCuts'] = ['SNR > %0.1f'%nemo_config['nemo']['snr']]
    template['makeTileDeck']  = mpi_switch

    if not mpi_switch:
        del template['tileDefinitions']
        del template['tileNoiseRegions']
        template['tileNoiseRegions'][0]['params']['noiseParams']['RADecSection'] = noise_tiles['0_0']
    else:
        template['tileDefinitions'] = tiles
        template['tileNoiseRegions'] = noise_tiles
        template['tileNoiseRegions'][0]['params']['noiseParams']['RADecSection'] = 'tileNoiseRegions'

    yaml.dump(template, open('{}.yaml'.format(patch), 'w'))

