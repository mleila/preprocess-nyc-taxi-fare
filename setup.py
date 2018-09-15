import os
from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='preprocess_nyc_taxi_fare',
    version='0.1',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    author='Mohamed Leila',
    description='A pandas-friendly preprocessing pipeline for the NYC taxi fare Kaggle competition dataset'
)
