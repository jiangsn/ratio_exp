'''
@Author: WANG Maonan
@Date: 2021-12-23 00:04:13
@Description: 
@LastEditTime: 2021-12-23 00:04:14
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='ratioReg',
    version=0.9,
    description='Predict Bar Ratio.',
    author='Maonan Wang',
    license='Apache License, Version 2.0',
    keywords='Ratio Regression',
    packages=find_packages(),
    install_requires=[i.strip() for i in open(os.path.join(os.path.dirname(__file__), 'requirements.txt')).readlines() if i.strip()],
    include_package_data=True,
)