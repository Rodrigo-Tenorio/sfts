"""
Linearly-chirping signal. This is a more general case of the BNS example
in which amplitude parameters are not marginalized out.
"""

import jax
import jax.numpy as jnp

from sfts import kernels, iphenot

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("This example needs matplotlib!") from e


def phase(times, phi_0, f_0, f_1):
    """
    Returns the instantaneous phase of a linear chirp.

    Parameters
    ----------
    times: (N,) array
        Timestamps at which the signal will be evaluated.

    phi_0, f_0, f_1: float
        Initial phase, frequency, and spindown

    Returns
    -------
    phase: (N,) array
    """
    return phi_0 + 2 * jnp.pi * (f_0 * times + 0.5 * f_1 * times**2)


def frequency(times, phi_0, f_0, f_1):
    return jax.vmap(jax.grad(phase), in_axes=(0, None, None, None), out_axes=0)(
        times, phi_0, f_0, f_1
    ) / (2 * jnp.pi)


# Generate data
amp = 2.0
phi_0 = jnp.pi / 3
f_0 = 5e-3
f_1 = 5e-11
deltaT = 1.0
duration = 1800

t_s = deltaT * jnp.arange(0, int(duration // deltaT))

data = amp * jnp.sin(phase(t_s, phi_0, f_0, f_1))

# Compute SFTs
P = 10
T_sft = 10.

samples_per_sft = jnp.floor(T_sft / deltaT).astype(int)
num_sfts = data.size // samples_per_sft
t_alpha = T_sft * jnp.arange(num_sfts)

## sfts: (frequency index, time index)
data_sfts = (
    deltaT
    * jnp.fft.rfft(
        data[: num_sfts * samples_per_sft].reshape(-1, samples_per_sft), axis=1
    ).T
)


# Compute scalar product
def scalar_product(A_alpha, phi_alpha, f_alpha, fdot_alpha):
    # Non-signal-dependent values are passed here by clousure
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

    return to_project.imag.sum()


# Evaluate the *vectorised* scalar product for a bunch of equal-mass systems
num_templates = 10_000
batch_size = 10
num_batches = int(num_templates // batch_size)

key = jax.random.key(992791)


def eval_templates(batch_ind, carry_on):

    key, out_vals, amp_temps, phi0_temps, f0_temps, f1_temps = carry_on

    key, key0, key1, key2, key3 = jax.random.split(key, 5)
    amps = amp + (2 * jax.random.uniform(key0, (batch_size,)) - 1)
    phi_0s = phi_0 + 0.1 * (2 * jax.random.uniform(key1, (batch_size,))- 1)
    f_0s = 4e-3 + 2e-3 * jax.random.uniform(key2, (batch_size,))
    f_1s = 4e-11 + 2e-11 * jax.random.uniform(key3, (batch_size,))

    A_alpha = amps[:, None]
    fdot_alpha = f_1s[:, None]
    phi_alpha = jax.vmap(
        phase,
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )(t_alpha, phi_0s, f_0s, f_1s)

    f_alpha = jax.vmap(frequency, in_axes=(None, 0, 0, 0), out_axes=0)(
        t_alpha, phi_0s, f_0s, f_1s
    )

    results = jax.vmap(scalar_product, in_axes=0, out_axes=0)(
        A_alpha, phi_alpha, f_alpha, fdot_alpha
    )

    out_vals = jax.lax.dynamic_update_slice(
        out_vals, results, (batch_ind * batch_size,)
    )
    amp_temps = jax.lax.dynamic_update_slice(amp_temps, amps, (batch_ind * batch_size,))
    phi0_temps = jax.lax.dynamic_update_slice(
        phi0_temps, phi_0s, (batch_ind * batch_size,)
    )
    f0_temps = jax.lax.dynamic_update_slice(f0_temps, f_0s, (batch_ind * batch_size,))
    f1_temps = jax.lax.dynamic_update_slice(f1_temps, f_1s, (batch_ind * batch_size,))

    return key, out_vals, amp_temps, phi0_temps, f0_temps, f1_temps


# Note that `fori_loop` on its own jit-compiles `eval_templates`,
# so no need to `jax.jit` anything so far.
out_vals = jnp.zeros(num_templates)
amp_temps = jnp.zeros(num_templates)
phi0_temps = jnp.zeros(num_templates)
f0_temps = jnp.zeros(num_templates)
f1_temps = jnp.zeros(num_templates)

print("Ready for the loop...")
(key, out_vals, amp_temps, phi0_temps, f0_temps, f1_temps) = jax.lax.fori_loop(
    0,
    num_batches,
    eval_templates,
    (key, out_vals, amp_temps, phi0_temps, f0_temps, f1_temps),
)

for label, true, temps in zip(
    ["f_0", "f_1", "amp", "phi0"],
    [f_0, f_1, amp, phi_0],
    [f0_temps, f1_temps, amp_temps, phi0_temps],
):
    X = temps - true

    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlabel=f"{label} -- (Template - True)")
    ax.plot(X, jnp.abs(out_vals), "o")
    fig.savefig(f"{label}.pdf")

sorting_keys = jnp.argsort(out_vals)

fig, ax = plt.subplots()
ax.set(xlabel="Amplitude", ylabel="Initial phase")
ax.scatter(amp_temps[sorting_keys], phi0_temps[sorting_keys], c=out_vals[sorting_keys])
ax.plot([amp], [phi_0], "*", color="black")
fig.savefig("amp_phi0.pdf")
