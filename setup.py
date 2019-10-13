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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="dory",
    package_dir={"dory": "dory"},
    install_requires=["astropy >= 3.2",
                    "numpy >= 1.10",
                    "matplotlib >= 2.0",
                    "astLib >= 0.10",
                    "pixell >= 0.5",
                    "scipy",
                    "PyYAML",
                    "mpi4py",
                    "pandas"],
    license="BSD license",
    keywords='dory',
    name='dory',
    packages=['dory'],
    url='https://github.com/dwhan89/dory',
)

print('\n[setup.py request was successful.]')

