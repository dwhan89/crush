from crush import config
from soapack import interfaces as soint
from pixell import enmap
import os

strict = False

nemo_config_file = config.package_data_path('configs/nemo.yaml')
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

    # start automatic tile generation
    if 'auto' in tile_setting:
        ivar = enmap.read_fits(weight_file)
print(patches)
print(nemo_config)
