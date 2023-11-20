"""
Setup script for c4_arxiv.

This script packages and distributes the associated wheel file(s).
Source code is in ./src/. Run 'python setup.py sdist bdist_wheel' to build.
"""
from setuptools import setup, find_packages

import sys
sys.path.append('./src')

import c4_arxiv

setup(
    name="c4_arxiv",
    version=c4_arxiv.__version__,
    url="https://databricks.com",
    author="xiaohan.zhang@databricks.com",
    description="my test wheel",
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    entry_points={"entry_points": "main=c4_arxiv.main:main"},
    install_requires=["setuptools"],
)
