#! /usr/bin/env python3
# -*- coding: utf8 -*-


import os
import sys
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "QRel",
    version = "0.1",
    author = "Florian Kunneman, Thiago Castro Ferreira",
    author_email = "f.kunneman@gmail.com",
    description = ("Framework for relating questions by their topics"),
    license = "GPL",
    keywords = "nlp computational_linguistics information_retrieval semantics",
    url = "https://github.com/fkunneman/QRel",
    packages=['qrel','qrel.functions', 'qrel.modules','qrel.classes'],
    #long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    zip_safe=False,
    include_package_data=True,
    package_data = {'': ['*.wsgi','*.js','*.xsl','*.gif','*.png','*.xml','*.html','*.jpg','*.svg','*.rng'] },
    setup_requires=['setuptools>=28.5.0'],
    install_requires=['gensim','numpy','sklearn','nltk'],
)
