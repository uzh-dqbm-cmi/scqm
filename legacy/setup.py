from os import path
from setuptools import setup, find_packages

# Project
NAME = 'scqm'
VERSION = '0.0.1'

# Authors and maintainers
AUTHORS = 'Aron Horvath'
MAINTAINER = 'Aron Horvath'
MAINTAINER_EMAIL = 'aronnorbert.horvath@uzh.ch'

# License
LICENSE = 'MIT'

# Project URLs
REPOSITORY = 'https://github.com/uzh-dqbm-cmi/scqm'
HOMEPAGE = 'https://github.com/uzh-dqbm-cmi/scqm'
PROJECT_URLS = {
    'Bug Tracker': f'{REPOSITORY}/issues',
    'Documentation': HOMEPAGE,
    'Source Code': REPOSITORY,
}
DOWNLOAD_URL = ''

# Classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: MIT License"
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "Framework :: Pytest",
    "Framework :: Flake8",
]

# Long description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Install requirements
with open(path.join(this_directory, 'requirements/production.txt'), encoding='utf-8') as f:
    INSTALL_REQUIREMENTS = f.read().splitlines()

# Package definition
setup(name=NAME,
      version=VERSION,
      description='Machine Learning tools for health care data analysis and prediction',
      url=HOMEPAGE,
      packages=find_packages(),
      author=AUTHORS,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      python_requires='>3.9.0',
      install_requires=INSTALL_REQUIREMENTS,
      include_package_data=True,
      zip_safe=False,
      )
