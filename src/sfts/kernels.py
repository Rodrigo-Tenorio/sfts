import jax.numpy as jnp
from jax.scipy import special

def discrete_dirichlet(k, N):
    """
    Dirichlet kernel using discrete variables.
    See Appendix of Tenorio & Gerosa (2025).

    Parameters
    ----------
    k:
        Frequency in ``index'' units (i.e. `floor(f / Tsft)`).
    N:
        Number of time-domain samples
    """
    return N * jnp.exp(1j * jnp.pi * k) * jnp.sinc(k)

def fresnel_kernel(f_0, f_1, T_sft):
    """
    Fresnel kernel.
    """
    quot = f_0 / f_1
    factor = jnp.sqrt(2 * f_1)

    Sl, Cl = special.fresnel(factor * quot)
    Su, Cu = special.fresnel(factor * (quot + T_sft))

    return jnp.exp(-1j * jnp.pi * f_0**2 / f_1) * ((Cu - Cl) + 1j * (Su - Sl)) / factor
