[![ReadTheDocs](https://img.shields.io/readthedocs/flowws-freud.svg?style=flat)](https://flowws-freud.readthedocs.io/en/latest/)

## Introduction

`flowws-freud` is an in-development set of modules to create reusable
pipelines for scientific simulations using
[freud](https://freud.readthedocs.io).

`flowws-freud` is being developed in conjunction with
[flowws](https://github.com/klarh/flowws) and
[flowws-analysis](https://github.com/klarh/flowws-analysis). See their
documentation for an overview of how to use the modules found here and
other useful modules for analysis and visualization of simulation
data, respectively.

## Installation

Install `flowws-freud` from PyPI:

```
pip install flowws-freud
```

Alternatively, install from source:

```
pip install git+https://github.com/klarh/flowws-freud.git#egg=flowws-freud
```

## Examples

Consult the
[flowws-examples](https://github.com/klarh/flowws-examples) project
for examples using `flowws-freud` modules.

## API Documentation

Browse more detailed documentation
[online](https://flowws-freud.readthedocs.io) or build the sphinx
documentation from source:

```
git clone https://github.com/klarh/flowws-freud
cd flowws-freud/doc
pip install -r requirements.txt
make html
```
