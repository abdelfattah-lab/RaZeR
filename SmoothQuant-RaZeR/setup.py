from struct import pack
from setuptools import setup, find_packages
setup(
    name='smoothquant',
    packages=find_packages(exclude=['act_scales'])
)
