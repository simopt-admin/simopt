"""Setup file for simopt package."""

from setuptools import setup, find_packages

setup(
    name='simopt',
    version='0.1.3',
    packages=find_packages(),
    entry_points={
        'gui_scripts': [
            'simopt = simopt.__main__:main'
        ]
    },
)