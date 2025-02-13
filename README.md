# `sfts`

Short Fourier Transforms for Fresnel-weighted Template Summation.

Implementation of gravitational-wave data-analysis tools described in [Tenorio & Gerosa (2025)][sfts]
to operate using Short Fourier Transforms (SFTs).

See [this simple example](./examples/bns_inspiral.py) for a quick-start on
using [`iphenot`][iphenot] (`/ˈaɪv ˈnɒt/`) and SFTs.

The package is composed of two main modules:

1. [iphenot.py][iphenot]: [jaxified](https://github.com/jax-ml/jax) re-implementation of the
inspiral part of the  [`IMRPhenomT` waveform approximant][LALPhenomT].
1. [kernels.py](./src/sfts/kernels.py): Fresnel and Dirichlet kernels to compute scalar products using SFTs.

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
@article{Tenorio:2025XYZ}
```
Whenever applicable, please consider also citing the `IMRPhenomT` papers [listed here][2].

[sfts]: https://arxiv.org
[LALPhenomT]: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM.c
[iphenot]: ./src/sfts/iphenot.py
