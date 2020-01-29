#!/usr/bin/env python

import os
from setuptools import setup

with open('flowws_freud/version.py') as version_file:
    exec(version_file.read())

module_names = [
    'RDF',
    'SmoothBOD',
]

flowws_modules = ['{0} = flowws_freud.{0}:{0}'.format(name) for name in module_names]

setup(name='flowws-freud',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows using freud',
      entry_points={
          'flowws_modules': flowws_modules,
      },
      extras_require={},
      install_requires=[
          'flowws-analysis',
          'freud-analysis',
      ],
      license='MIT',
      packages=[
          'flowws_freud',
      ],
      python_requires='>=3',
      version=__version__
      )
