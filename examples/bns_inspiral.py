"""
Simulate the inspiral of a (1.4 - 1.4) BNS from 5 Hz to 10 Hz
and runs a small template bank for different massess as an example
application of sfts.

This is just an illustrative example to show the way of interfacing
`sfts` with `jax`'s primitives. Do not assume this code operates
following any particular waveform conventions.
"""

import jax
import jax.numpy as jnp

from sfts import kernels, iphenot

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("This example needs matplotlib!") from e


def generate_waveform(times, coeffs):
    """
    Returns the amplitude, phase, frequency, and frequency derivative
    of a *single* waveform with coefficients `coeffs` as generated
    by `iphenot.bunch_of_coeffs` for *an array* of timestamps `times`.


    Parameters
    ----------
    times: (N,) array
        Timestamps at which the waveform will be evaluated.
        Must use `iphenot.t_of_f` to find those corresponding to
        the desired frequency.

    coeffs: dict
        Coefficients as returned by `iphenot.bunch_of_coeffs`

    Returns
    -------
    amp, phase, freq, fdot: (N,) array
    """

    # sfts.iphenot functions are to be vectorised on demand.
    # Here, we use `jax.vmap` to vectorise the time evaluation:
    phase = jax.vmap(iphenot.phi_of_t, in_axes=(0, None), out_axes=0)(times, coeffs)

    # Similarly, fdot can be obtained using jax's autodiff:
    frequency, fdot = jax.vmap(
        jax.value_and_grad(iphenot.f_of_t), in_axes=(0, None), out_axes=0
    )(times, coeffs)

    amp = jax.vmap(iphenot.amp_of_t, in_axes=(0, None), out_axes=0)(frequency, coeffs)

    return amp, phase, frequency, fdot


# Generate data
min_freq = 5.0
max_freq = 10.0
deltaT = 1 / 40.0

data_coeffs = iphenot.bunch_of_coeffs(
    1.4 * iphenot.MSUN_SI, 1.4 * iphenot.MSUN_SI, chi1L=0, chi2L=0, distance=1e3
)

t_min = iphenot.t_of_f(min_freq, data_coeffs, check_bracket=True)
t_max = iphenot.t_of_f(max_freq, data_coeffs, check_bracket=True)
t_s = t_min + deltaT * jnp.arange(jnp.ceil((t_max - t_min) / deltaT))

amp_d, phase_d, _, _ = generate_waveform(t_s, data_coeffs)

## The projector in this case is simply (Re + Im)
data = amp_d * (jnp.cos(phase_d) + jnp.sin(phase_d))

# Compute SFTs
delta = 1
P = 50  # Deltak / 2
max_fdotdot = jax.grad(jax.grad(iphenot.f_of_t))(t_max, data_coeffs)
T_sft = jnp.cbrt(2 * delta / max_fdotdot).astype(int)

samples_per_sft = jnp.floor(T_sft / deltaT).astype(int)
num_sfts = data.size // samples_per_sft
t_alpha = t_min + T_sft * jnp.arange(num_sfts)

## sfts: (frequency index, time index)
data_sfts = (
    deltaT
    * jnp.fft.rfft(
        data[: num_sfts * samples_per_sft].reshape(-1, samples_per_sft), axis=1
    ).T
)


# Compute scalar product
def scalar_product(A_alpha, phi_alpha, f_alpha, fdot_alpha):
    # Non-waveform-dependent values are passed here by clousure
    deltaf = 1 / T_sft

    f_k_of_alpha = (f_alpha * T_sft).astype(int)
    k_min_max = f_k_of_alpha + jnp.arange(-P, P + 1)[:, None]

    # Set to 0 whatever gets beyond the range.
    # Note that jax will not complain about out-of-range indexing
    zero_mask = jnp.logical_or(k_min_max >= 0, k_min_max < data_sfts.shape[0])

    c_alpha = (
        deltaf
        * data_sfts[k_min_max, jnp.arange(num_sfts)].conj()
        * kernels.fresnel_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
        * zero_mask
    )

    to_project = A_alpha * jnp.exp(1j * phi_alpha) * c_alpha.sum(axis=0)

    return (to_project.real + to_project.imag).sum()


# Evaluate the *vectorised* scalar product for a bunch of equal-mass systems
num_templates = 10_000
batch_size = 10
num_batches = int(num_templates // batch_size)

key = jax.random.key(992791)


def eval_templates(batch_ind, carry_on):

    m1s_sun = 1.3 + 0.2 * jax.random.uniform(
        jax.random.fold_in(key, batch_ind), (batch_size,)
    )

    # Evaluate `coeffs` for the whole batch in parallel.
    coeffs = jax.vmap(iphenot.bunch_of_coeffs, in_axes=0, out_axes=0)(
        m1s_sun * iphenot.MSUN_SI,
        m1s_sun * iphenot.MSUN_SI,
        jnp.zeros_like(m1s_sun),
        jnp.zeros_like(m1s_sun),
        1e3 * jnp.ones_like(m1s_sun),
    )

    # Further vmap `generate_waveform` to parallelize over templates.
    # Note that we don't care about the innter details of `generate_waveform`,
    # those are for `jax` to figure out
    A_alpha, phi_alpha, f_alpha, fdot_alpha = jax.vmap(
        generate_waveform, in_axes=(None, 0), out_axes=0
    )(t_alpha, coeffs)

    # Finally, vmap `scalar_product` to evaluate the whole batch atonce
    results = jax.vmap(scalar_product, in_axes=0, out_axes=0)(
        A_alpha, phi_alpha, f_alpha, fdot_alpha
    )

    # This is how you fill up an array with a moving starting index in jax.
    out_vals, m1_templates = carry_on
    out_vals = jax.lax.dynamic_update_slice(
        out_vals, results, (batch_ind * batch_size,)
    )
    m1_templates = jax.lax.dynamic_update_slice(
        m1_templates, m1s_sun, (batch_ind * batch_size,)
    )

    return (out_vals, m1_templates)


# Note that `fori_loop` on its own jit-compiles `eval_templates`,
# so no need to `jax.jit` anything so far.
out_vals = jnp.zeros(num_templates)
m1_templates = jnp.zeros(num_templates)

print("Ready for the loop...")
(out_vals, m1_templates) = jax.lax.fori_loop(
    0, num_batches, eval_templates, (out_vals, m1_templates)
)

on_target = scalar_product(*generate_waveform(t_alpha, data_coeffs))

fig, ax = plt.subplots()
ax.set(xlabel="NS mass (equal-mass system) in Solar massess", ylabel="Scalar product")
ax.plot(m1_templates, out_vals, "o")
ax.axvline(1.4, color="red", ls="--")
ax.plot([1.4], [on_target], "d", color="orange")
plt.show()
