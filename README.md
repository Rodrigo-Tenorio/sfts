# `sfts`

[![arXiv](https://img.shields.io/badge/arXiv-2502.11823-b31b1b.svg)](https://arxiv.org/abs/2502.11823)
[![DOI](https://zenodo.org/badge/931409818.svg)](https://zenodo.org/badge/latestdoi/931409818)
[![PyPI version](https://badge.fury.io/py/sfts.svg)](https://badge.fury.io/py/sfts)


Short Fourier Transforms for Fresnel-weighted Template Summation.

Implementation of gravitational-wave data-analysis tools described in [Tenorio & Gerosa (2025)][sfts]
to operate using Short Fourier Transforms (SFTs).

See the [examples](./examples) for a quick-start on using SFTs and [`iphenot`][iphenot] (`/ˈaɪv ˈnɒt/`).

`sfts` contains two main modules:

1. [kernels.py](./src/sfts/kernels.py): Fresnel and Dirichlet kernels to compute scalar products using SFTs.
1. [iphenot.py][iphenot]: [jaxified](https://github.com/jax-ml/jax) re-implementation of the
inspiral part of the  [`IMRPhenomT` waveform approximant][LALPhenomT].

The SFT convention in this package is compatible with that in the
[LVK  `.sft` file format](https://dcc.ligo.org/LIGO-T040164/public). 
Checkout [fasttracks](https://github.com/Rodrigo-Tenorio/sfts)'
[search example](https://github.com/Rodrigo-Tenorio/fasttracks/blob/master/examples/run_binary_search.py) 
to learn about reading `.sft` files into `jax` arrays.

# How to install

`sfts` can be pulled in from PyPI:
```
$ pip install sfts
```

To pull in `jax`'s GPU capabilities, use:

```
$ pip install sfts[cuda]
```

Alternatively, this repository itself is pip-installable.

# Cite

If the tools provided by `sfts` were useful to you, we would appreciate a citation of
[the accompanying paper][sfts]:
```
@article{Tenorio:2025yca,
    author = "Tenorio, Rodrigo and Gerosa, Davide",
    title = "{SFTs: a scalable data-analysis framework for long-duration gravitational-wave signals}",
    eprint = "2502.11823",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "2",
    year = "2025"
}
```
Whenever applicable, please consider also citing the `IMRPhenomT` papers [listed here][LALPhenomT].

[sfts]: https://arxiv.org/abs/2502.11823
[LALPhenomT]: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM.c
[iphenot]: ./src/sfts/iphenot.py
