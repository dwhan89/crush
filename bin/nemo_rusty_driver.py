import os
from string import Template
from subprocess import Popen, PIPE

import numpy as np
import yaml
from pixell import enmap, utils
from soapack import interfaces as soint

from crush import config, maps

strict = True
overwrite = False
submit_jobs = True

nemo_config_file = config.package_data_path('configs/nemo.yaml')
template_file = config.package_data_path('configs/nemo_template.yaml')
nemo_config = config.read_yaml(nemo_config_file)

# load data model
DM = soint.models['act_mr3']()

patches = nemo_config['nemo']['patches']
if type(patches) != list:
    assert (patches in soint.models.keys())  # make sure data model is available
    # load all available patches in that model
    patches = DM.get_psa_indexes()
else:
    pass

for psa in patches:
    print("processing %s" % psa)
    template = config.read_yaml(template_file)

    season, patch, array, freq = psa.split('_')
    if patch not in ['cmb']: continue
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

    ngrids = [1, 1]
    # start automatic tile generation
    noise_tiles = {'autoBorderDeg': 0.5}
    tiles = []
    if 'auto' in tile_setting:
        ivar = enmap.read_fits(weight_file)
        shape, wcs = ivar.shape, ivar.wcs
        default_extent = np.array(nemo_config['nemo']['default_tile_extent']) * utils.degree
        ngrids = maps.nrect_grid(ivar, default_extent)
        nygrid, nxgrid = ngrids

        # some simplification
        if nxgrid * nygrid <= 8:
            ngrids = [1, 1]
            nxgrid = nygrid = 1

        grid_pix = maps.rect_grid_pix(shape, ngrids)
        valid_grid = maps.threshold_grids(ivar, grid_pix)

        # put this in the setting!
        threshold_factor = 1.3 if patch not in ['cmb', 'boss'] else 0.8

        noise_grid_pix = maps.bounded_pixs(ivar, grid_pix, valid_grid, threshold_factor=threshold_factor, sigma=2,
                                           downsample=10)

        grid_coords = maps.gridpix2sky(shape, wcs, grid_pix) / utils.degree
        noise_grid_coords = maps.gridpix2sky(shape, wcs, noise_grid_pix)
        noise_grid_coords = maps.reguarlize_rect_grid(noise_grid_coords / utils.degree)
        for j in np.arange(nygrid):
            for i in np.arange(nxgrid):
                if not valid_grid[j, i]: continue
                decs, dece, ras, rae = grid_coords[j, i].tolist()
                tiles.append({'tileName': "{}_{}".format(j, i), 'RADecSection': [ras, rae, decs, dece]})
                decs, dece, ras, rae = noise_grid_coords[j, i].tolist()
                noise_tiles["{}_{}".format(j, i)] = [ras, rae, decs, dece]
    if 'custom' in tile_setting:
        # do something here
        pass

    mpi_switch = (ngrids[0] * ngrids[1] != 1)
    mpi_switch = True if mpi_switch else False
    template['unfilteredMaps'][0]['mapFileName'] = map_file
    template['unfilteredMaps'][0]['weightsFileName'] = weight_file
    template['unfilteredMaps'][0]['beamFileName'] = beam_file
    template['unfilteredMaps'][0]['obsFreqGHz'] = 95.0 if freq == 'f090' else 148.0
    template['useMPI'] = mpi_switch
    template['thresholdSigma'] = nemo_config['nemo']['snr']
    template['objIdent'] = 'mr3c_{}-'.format(psa)
    template['catalogCuts'] = ['SNR > %0.1f' % nemo_config['nemo']['snr']]
    template['makeTileDir'] = mpi_switch

    if not mpi_switch:
        del template['tileDefinitions']
        del template['tileNoiseRegions']
        template['mapFilters'][0]['params']['noiseParams']['RADecSection'] = noise_tiles['0_0']
    else:
        template['tileDefinitions'] = tiles
        template['tileNoiseRegions'] = noise_tiles
        template['mapFilters'][0]['params']['noiseParams']['RADecSection'] = 'tileNoiseRegions'

    yaml.dump(template, open('{}.yml'.format(psa), 'w'))

    if not submit_jobs: continue
    slurm_template_file = open(config.package_data_path('configs/rusty_slurm.txt'), 'r')
    slurm_template = Template(slurm_template_file.read())
    slurm_template_file.close()
    ngrid = int(ngrids[0] * ngrids[1])

    # improve this part
    nodes = int(np.ceil(ngrid / 40))
    ntasks = ngrid
    slurm_settings = {}
    slurm_settings['nodes'] = nodes
    slurm_settings['ntasks'] = ntasks
    slurm_settings['times'] = '24:00:00'
    slurm_settings['cpuspertask'] = 1
    slurm_settings['jobname'] = psa
    slurm_settings['nemo_yml'] = '{}.yml'.format(psa)
    slurm_settings['nemo_flags'] = '-M' if mpi_switch else ''
    slurm_settings['queue'] = 'gen'

    script_content = slurm_template.safe_substitute(slurm_settings)

    script = open('batch.txt', 'w')
    script.write(script_content)
    script.close()

    command = ['sbatch', 'batch.txt']
    process = Popen(command, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    os.remove('batch.txt')
