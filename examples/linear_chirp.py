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


key = jax.random.key(992791)
P = 50
T_sft = 10 * 86400.0

# Generate data
amp = 1.
phi_0 = jnp.pi / 3
f_0 = 4.0
f_1 = 5e-11
deltaT = 1 / (10.0)
duration = 100 * 86400.0

t_s = deltaT * jnp.arange(0, int(duration // deltaT))

key, subkey = jax.random.split(key)
data = amp * jnp.sin(phase(t_s, phi_0, f_0, f_1)) + 1 * jax.random.normal(
    key, t_s.shape
)

# Paranoia checks

df0 = 1 / T_sft
df1 = 1 / (T_sft * duration)

max_freq = f_0 + f_1 * duration
fsamp = 2 / (deltaT)
drift_bins = T_sft * f_1 / df0

print(f"Maximum frequency: {max_freq} Hz")
print(f"Sampling rate: {fsamp} Hz")
print(f"SFT freq bins: {0.5 * T_sft // deltaT}")
print(f"Drift bins per SFT: {drift_bins}")

if drift_bins > P:
    print("WARNING: P is too small given how many bins this signal drifts")

if max_freq > fsamp:
    raise ValueError(f"Maximum frequency {max_freq} too high for sampling rate {fsamp}")

print(f"SFT frequency resolution: {df0:.2g} Hz")
print(f"f_0 = {f_0} Hz = {f_0 / df0} bins")
print(f"Spindown resolution: {df1:.2g} Hz/s")
print(f"f_1 = {f_1} Hz/s = {f_1 / df1} bins")


# Compute SFTs
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
num_templates = 10000
batch_size = 100
num_batches = int(num_templates // batch_size)


def eval_templates(batch_ind, carry_on):

    key, out_vals, amp_temps, phi0_temps, f0_temps, f1_temps = carry_on
    bins_per_dim = 0.5

    key, key0, key1, key2, key3 = jax.random.split(key, 5)
    amps = amp + 0 * jax.random.uniform(
        key0,
        (batch_size,),
        minval=-1,
        maxval=1,
    )
    phi_0s = phi_0 + jax.random.uniform(
        key1, (batch_size,), minval=-1e-3/duration, maxval=1e-3/duration
    )
    f_0s = f_0 + df0 * jax.random.uniform(
        key2, (batch_size,), minval=-bins_per_dim, maxval=bins_per_dim
    )
    f_1s = f_1 + df1 * jax.random.uniform(
        key3, (batch_size,), minval=-bins_per_dim, maxval=bins_per_dim
    )

    A_alpha = amps[:, None]
    fdot_alpha = f_1s[:, None]

    phi_alpha = jax.vmap(
        phase,
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )(t_alpha, phi_0s, f_0s, f_1s)
    f_alpha = f_0s[:, None] + t_alpha * f_1s[:, None]

    results = jnp.abs(
        jax.vmap(scalar_product, in_axes=0, out_axes=0)(
            A_alpha, phi_alpha, f_alpha, fdot_alpha
        )
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

sorting_keys = jnp.argsort(out_vals)

for label, true, temps in zip(
    ["f_0", "f_1", "amp", "phi0"],
    [f_0, f_1, amp, phi_0],
    [f0_temps, f1_temps, amp_temps, phi0_temps],
):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlabel=f"{label}")
    ax.plot(temps, jnp.abs(out_vals), "o")
    ax.axvline(true, ls="--", color="red")
    fig.savefig(f"{label}.pdf")

    if label != "f_0":
        fig, ax = plt.subplots()
        ax.set(xlabel="f0 [Hz]", ylabel=label)
        c = ax.scatter(
            f0_temps[sorting_keys],
            temps[sorting_keys],
            c=out_vals[sorting_keys],
            cmap="plasma",
        )
        ax.plot([f_0], [true], "*", color="black", markerfacecolor="none", markersize=10)
        fig.colorbar(c)
        fig.savefig(f"f_0_{label}.pdf")
