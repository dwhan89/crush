import os
import warnings

import yaml

_root = os.path.abspath(os.path.dirname(__file__))


def package_data_path(fname, strict=False):
    file_path = os.path.join(_root, 'data', fname)
    if not os.path.exists(file_path):
        if strict:
            raise FileNotFoundError("Can't find %s" % file_path)
        else:
            warnings.warn("Can't find %s" % file_path)
    else:
        pass
    return file_path


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
