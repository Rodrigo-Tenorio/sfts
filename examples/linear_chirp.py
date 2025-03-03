"""
Linearly-chirping signal. This is a more general case of the BNS example
in which amplitude parameters are not marginalized out.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sfts import kernels

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


key = jax.random.key(12322)

# Generate data
amp = 25.0
phi_0 = jnp.pi / 3
f_0 = 1.0
f_1 = 1e-12
deltaT = 1 / 4.0
duration = 1000 * 86400.0

P = 1000
T_sft = 100 * 86400.0

t_s = deltaT * jnp.arange(int(duration // deltaT))

key, subkey = jax.random.split(key)
data = amp * jnp.sin(phase(t_s, phi_0, f_0, f_1))

dd_term = (deltaT * data**2).sum()

# Paranoia checks

df0 = 1 / T_sft
df1 = 1 / (T_sft * duration)

max_freq = f_0 + f_1 * duration
fsamp = 1.0 / (deltaT)
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
# [See Eq. (7) of Tenorio & Gerosa 2025]
def det_stat(A_alpha, phi_alpha, f_alpha, fdot_alpha):
    # Non-signal-dependent values are passed here by clousure
    deltaf = 1 / T_sft

    f_k_of_alpha = (f_alpha * T_sft).astype(int)
    k_min_max = f_k_of_alpha + jnp.arange(-P, P + 1)[:, None]

    # Set to 0 whatever gets beyond the range.
    # Note that jax will not complain about out-of-range indexing
    zero_mask = jnp.logical_or(k_min_max >= 0, k_min_max < data_sfts.shape[0])

    c_alpha = (
        deltaf
        * data_sfts[k_min_max, jnp.arange(data_sfts.shape[1])].conj()
        * kernels.fresnel_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
        * zero_mask
    )
    dh_term = (A_alpha * (jnp.exp(1j * phi_alpha) * c_alpha.sum(axis=0)).imag).sum()
    hh_term = 0.5 * T_sft * (A_alpha**2).sum()

    return dh_term - 0.5 * (dd_term + hh_term)


# Evaluate the *vectorised* scalar product for a bunch of linear chirps
num_templates = 10000
batch_size = 100
num_batches = int(num_templates // batch_size)


def eval_templates(batch_ind, carry_on):

    key, out_vals = carry_on

    key, key0, key1, key2, key3 = jax.random.split(key, 5)
    f_0s = f_0 + (1 / duration) * jax.random.uniform(
        key0, (batch_size,), minval=-2, maxval=2
    )
    f_1s = f_1 + (1 / duration**2) * jax.random.uniform(
        key1, (batch_size,), minval=-0.5, maxval=0.5
    )
    amps = jax.random.uniform(key2, (batch_size,), minval=10, maxval=35)
    phi_0s = phi_0 + jax.random.uniform(
        key3, (batch_size,), minval=-phi_0, maxval=2 * jnp.pi - phi_0
    )

    # Technically too general, but safer when doing sums
    A_alpha = amps[:, None] * jnp.ones_like(t_alpha)
    fdot_alpha = f_1s[:, None] * jnp.ones_like(t_alpha)

    phi_alpha = jax.vmap(
        phase,
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )(t_alpha, phi_0s, f_0s, f_1s)

    f_alpha = jax.vmap(
        jax.vmap(jax.grad(phase), in_axes=(0, None, None, None), out_axes=0),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )(t_alpha, phi_0s, f_0s, f_1s) / (2 * jnp.pi)

    stats = jax.vmap(det_stat, in_axes=0, out_axes=0)(
        A_alpha, phi_alpha, f_alpha, fdot_alpha
    )

    results = jnp.c_[f_0s, f_1s, amps, phi_0s, stats]

    out_vals = jax.lax.dynamic_update_slice_in_dim(
        out_vals, results, batch_ind * batch_size, axis=0
    )

    return key, out_vals


# Note that `fori_loop` on its own jit-compiles `eval_templates`,
# so no need to `jax.jit` anything so far.
header = ["f_0", "f_1", "amp", "phi_0", "stat"]
out_vals = jnp.zeros((num_templates, 5))

print("Ready for the loop...")
(key, out_vals) = jax.lax.fori_loop(
    0,
    num_batches,
    eval_templates,
    (key, out_vals),
)

sorting_keys = jnp.argsort(out_vals[:, -1])

for ind, (label, true) in enumerate(
    zip(
        ["f_0", "f_1", "amp", "phi_0"],
        [f_0, f_1, amp, phi_0],
    )
):

    temps = out_vals[:, ind]

    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlabel=f"{label}")
    ax.plot(temps, out_vals[:, -1], "o")
    ax.axvline(true, ls="--", color="red")
    fig.savefig(f"{label}.pdf")

    if label != "f_0":
        fig, ax = plt.subplots()
        ax.set(xlabel="f_0 [Hz]", ylabel=label)
        c = ax.scatter(
            out_vals[sorting_keys, 0],
            temps[sorting_keys],
            c=out_vals[sorting_keys, -1],
            cmap="plasma",
        )
        ax.plot(
            [f_0], [true], "*", color="black", markerfacecolor="none", markersize=10
        )
        fig.colorbar(c)
        fig.savefig(f"f_0_{label}.pdf")

fig, ax = plt.subplots()
ax.set(xlabel="amp", ylabel="phi_0")
c = ax.scatter(
    out_vals[sorting_keys, 2],
    out_vals[sorting_keys, 3],
    c=out_vals[sorting_keys, -1],
    cmap="plasma",
)
ax.plot([amp], [phi_0], "*", color="black", markerfacecolor="none", markersize=10)
fig.colorbar(c)
fig.savefig("amp_phi_0.pdf")
