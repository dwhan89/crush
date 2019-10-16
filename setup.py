#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools
from distutils.errors import DistutilsError
from numpy.distutils.core import setup, Extension, build_ext, build_src
from distutils.sysconfig import get_config_var, get_config_vars
import versioneer
import os, sys
import subprocess as sp
import numpy as np
build_ext = build_ext.build_ext
build_src = build_src.build_src

setup(
    author="Dongwon 'DW' Han",
    author_email='dwhan89@gmail.com',
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Intended Audience :: Mostly Me',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="crush",
    package_dir={"crush": "crush"},
    install_requires=["astropy >= 3.2",
                    "numpy >= 1.10",
                    "matplotlib >= 2.0",
                    "astLib >= 0.10",
                    "pixell >= 0.5",
                    "scipy",
                    "PyYAML",
                    "mpi4py",
                    "pandas",
                    "soapack"],
    license="BSD license",
    keywords='crush',
    name='crush',
    packages=['crush'],
    include_package_data=True,
    package_data={'crush': ['data/*', 'data/misc/*']},
    url='https://github.com/dwhan89/crush',
)

print('\n[setup.py request was successful.]')

