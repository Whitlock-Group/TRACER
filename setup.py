# Copyright 2020-2021 The tracer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Install script for setuptools."""

from distutils import cmd
from distutils import log
import fnmatch
import os
import platform
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command import install
from setuptools.command import test

PLATFORM_SUFFIXES = {
    'Linux': 'linux',
    'Windows': 'win64',
    'Darwin': 'macos',
}



# def find_data_files(package_dir, patterns, excludes=()):
#   """Recursively finds files whose names match the given shell patterns."""
#   paths = set()
#
#   def is_excluded(s):
#     for exclude in excludes:
#       if fnmatch.fnmatch(s, exclude):
#         return True
#     return False
#
#   for directory, _, filenames in os.walk(package_dir):
#     if is_excluded(directory):
#       continue
#     for pattern in patterns:
#       for filename in fnmatch.filter(filenames, pattern):
#         # NB: paths must be relative to the package directory.
#         relative_dirpath = os.path.relpath(directory, package_dir)
#         full_path = os.path.join(relative_dirpath, filename)
#         if not is_excluded(full_path):
#           paths.add(full_path)
#   return list(paths)


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tracer",
    version="2.0.0",
    author="TRACER",
    description='Reconstructing Anatomical Coordinates in Rats in 3 Dimensions.',
    license='Apache License, Version 2.0',
    keywords="Waxholm atlas probe tracing visualization 3d",
    python_requires='>=3.8',
    url='https://github.com/Whitlock-Group/TRACER',
    packages=['tracer'],
    long_description=read('README.md'),
    install_requires=[
        'numpy>=1.19.4',
        'matplotlib==3.3.3',
        'mplcursors>=0.3',
        'future',
        'Pillow>=8.1.0',
        'keyboard>=0.13.5',
        'scipy>=1.6.0',
        'nibabel>=3.2.1',
        'nilearn>=0.7.0',
        'opencv-python>=4.5.1.48',
        'scikit-image>=0.18.1',
        'scikit-spatial>=5.1.0',
        'vedo',
        'setuptools!=50.0.0',  # https://github.com/pypa/setuptools/issues/2350
        'scipy',
        'tabulate>=0.8.7',
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
