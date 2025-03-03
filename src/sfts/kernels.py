import jax.numpy as jnp
from jax.scipy import special

def dirichlet_kernel(f, T_sft):
    """
    Dirichlet kernel.
    See Appendix of Tenorio & Gerosa (2025).
    """
    return T_sft * jnp.exp(1j * jnp.pi * f * T_sft) * jnp.sinc(f * T_sft)

def fresnel_kernel(f_0, f_1, T_sft):
    """
    Fresnel kernel.
    See Eq. (29) of Tenorio & Gerosa (2025).
    """
    quot = f_0 / f_1
    factor = jnp.sqrt(2 * f_1)

    Sl, Cl = special.fresnel(factor * quot)
    Su, Cu = special.fresnel(factor * (quot + T_sft))

    return jnp.exp(-1j * jnp.pi * f_0**2 / f_1) * ((Cu - Cl) + 1j * (Su - Sl)) / factor
