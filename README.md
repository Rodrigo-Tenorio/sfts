# `sfts`

Short Fourier Transforms for Fresnel-weighted Template Summation

Implementation of gravitational-wave data-analysis tools described in [Tenorio & Gerosa (2025)][1]
to operate using Short Fourier Transforms (SFTs).

1. [kernels.py](./src/sfts/kernels.py): Fresnel and Dirichlet kernels to compute scalar products using SFTs.
2. [waveform.py](./src/iphenot/waveform.py): [jaxified](https://github.com/jax-ml/jax) re-implementation of the
inspiral part of the 
[`IMRPhenomT` waveform approximant][2].

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

If `iphenot` was useful to you, we would appreciate a citation of [the accompanying paper][1]:
```
@article{Tenorio:2025XYZ}
```
Whenever applicable, please consider citing the `IMRPhenomT` papers [listed here][2].

[1]: https://arxiv.org
[2]: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM.c
