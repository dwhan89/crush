from crush import config
import yaml

nemo_config_file = config.package_data_path('configs/nemo.yaml')
nemo_config = config.read_yaml(nemo_config_file)

print(nemo_config)