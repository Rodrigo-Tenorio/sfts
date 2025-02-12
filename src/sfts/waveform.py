"""
Exposes the main components of the waveform approximant:
    - Amplitude: `amp_of_t`
    - Phase: `phi_of_t`
    - Frequency: `f_of_t`
"""

import jax

# This is unfortunately a must
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.lax import integer_pow
from jaxopt import Bisection

from . import ansaetze, taylor_t3

# Constants taken from lal
MRSUN_SI = 1.476625038050124729627979840144936351e3
MSUN_SI = 1.988409870698050731911960804878414216e30
MTSUN_SI = 4.925490947641266978197229498498379006e-6
PC_SI = 3.085677581491367278913937957796471611e16


def phi_of_t(t_s, coeffs):
    thetabar = jax.lax.pow(-coeffs["eta"] * t_s / coeffs["M_sec"], -1 / 8.0)
    return ansaetze.IMRPhenomTInspiralPhaseAnsatz22(
        t_s / coeffs["M_sec"], thetabar, coeffs
    )


def f_of_t(t_s, coeffs):
    thetabar = jax.lax.pow(-coeffs["eta"] * t_s / coeffs["M_sec"], -1 / 8.0)
    theta = thetabar * jax.lax.pow(5.0, 1.0 / 8)
    return (
        ansaetze.IMRPhenomTInspiralOmegaAnsatz22(theta, coeffs)
        / (2 * jnp.pi)
        / coeffs["M_sec"]
    )


def amp_of_t(freq_of_t, coeffs):
    x = (0.5 * freq_of_t * 2 * jnp.pi * coeffs["M_sec"]) ** (2.0 / 3.0)
    return coeffs["ampfac"] * jnp.abs(ansaetze.IMRPhenomTInspiralAmpAnsatzHM(x, coeffs))


def t_of_f(f, coeffs, lower_s=-3e7, upper_s=-1e-2, check_bracket=False, tol=1e-8):
    return (
        Bisection(
            lambda t: f - f_of_t(t, coeffs),
            lower=lower_s,
            upper=upper_s,
            check_bracket=check_bracket,
            tol=tol,
        )
        .run()
        .params
    )


def bunch_of_coeffs(
    m1_SI,
    m2_SI,
    chi1L,
    chi2L,
    distance,
):
    """
    IMRPhenomTSetPhase22Coefficients
    """

    dchi = chi1L - chi2L

    m1 = m1_SI / MSUN_SI
    m2 = m2_SI / MSUN_SI
    Mtot = m1 + m2

    ampfac = Mtot * MRSUN_SI / distance

    delta = jnp.abs(m1 - m2) / (m1 + m2)
    eta = 0.25 * jnp.abs(1 - delta**2)
    S = (m1**2 * chi1L + m2**2 * chi2L) / (m1**2 + m2**2)

    # Inspiral stuff
    t0 = taylor_t3.t0(eta, S, dchi, delta)
    omega_n_PN_coeffs = taylor_t3.omega_n_PN(eta, delta, chi1L, chi2L)

    theta_points = jnp.array([0.33, 0.45, 0.55, 0.65, 0.75, 0.82])

    tEarly = -5.0 / (eta * integer_pow(theta_points[0], 8))
    thetaini = jax.lax.pow(eta * (t0 - tEarly) / 5, -1.0 / 8)

    omegainspoints = jnp.zeros(6)
    omegainspoints = omegainspoints.at[0].set(
        taylor_t3.phenom_Taylor_T3(thetaini, omega_n_PN_coeffs)
    )
    omegainspoints = omegainspoints.at[1].set(
        ansaetze.IMRPhenomT_Inspiral_Freq_CP1_22(eta, S, dchi, delta)
    )
    omegainspoints = omegainspoints.at[2].set(
        ansaetze.IMRPhenomT_Inspiral_Freq_CP2_22(eta, S, dchi, delta)
    )
    omegainspoints = omegainspoints.at[3].set(
        ansaetze.IMRPhenomT_Inspiral_Freq_CP3_22(eta, S, dchi, delta)
    )
    omegainspoints = omegainspoints.at[4].set(
        ansaetze.IMRPhenomT_Inspiral_Freq_CP4_22(eta, S, dchi, delta)
    )
    omegainspoints = omegainspoints.at[5].set(
        ansaetze.IMRPhenomT_Inspiral_Freq_CP5_22(eta, S, dchi, delta)
    )

    A = theta_points[:, None] ** jnp.arange(8, 14)
    b = (
        4
        / integer_pow(theta_points, 3)
        * (
            omegainspoints
            - jax.vmap(taylor_t3.phenom_Taylor_T3, in_axes=(0, None), out_axes=0)(
                theta_points, omega_n_PN_coeffs
            )
        )
    )
    omega_InspC = jnp.linalg.solve(A, b)
    tCut = -5.0 / (eta * integer_pow(0.81, 8))

    inspiral_coeffs = {  # These will be needed for amplitude fits
        "ampfac": ampfac,
        "eta": eta,
        "M_sec": Mtot * MTSUN_SI,
        "omega_n_PN_coeffs": omega_n_PN_coeffs,
        "omega_InspC_coeffs": omega_InspC,
        "tEarly": tEarly,
        "t0": t0,
        "tCut": tCut,
    }

    # Amplitude stuff, focusing only on 22
    fac0 = 2 * eta * jnp.sqrt(16 * jnp.pi / 5)
    ampN = 1.0

    ampPNreal = jnp.array(
        [
            0,
            -2.5476190476190474 + (55.0 * eta) / 42.0,
            (-2 * chi1L) / 3.0
            - (2 * chi2L) / 3.0
            - (2 * chi1L * delta) / (3.0 * ((1 - delta) / 2.0 + (1 + delta) / 2.0))
            + (2 * chi2L * delta) / (3.0 * ((1 - delta) / 2.0 + (1 + delta) / 2.0))
            + (2 * chi1L * eta) / 3.0
            + (2 * chi2L * eta) / 3.0
            + 2 * jnp.pi,
            -1.437169312169312
            + integer_pow(chi1L, 2) / 2.0
            + integer_pow(chi2L, 2) / 2.0
            + (integer_pow(chi1L, 2) * delta) / 2.0
            - (integer_pow(chi2L, 2) * delta) / 2.0
            - (1069.0 * eta) / 216.0
            - integer_pow(chi1L, 2) * eta
            + 2 * chi1L * chi2L * eta
            - integer_pow(chi2L, 2) * eta
            + (2047.0 * integer_pow(eta, 2)) / 1512.0,
            -(107.0 * jnp.pi) / 21.0 + (34 * eta * jnp.pi) / 21.0,
            41.78634662956092
            - (278185.0 * eta) / 33264.0
            - (20261.0 * integer_pow(eta, 2)) / 2772.0
            + (114635.0 * integer_pow(eta, 3)) / 99792.0
            - (856.0 * jnp.euler_gamma) / 105.0
            + (2.0 * integer_pow(jnp.pi, 2)) / 3.0
            + (41.0 * eta * integer_pow(jnp.pi, 2)) / 96.0,
            (-2173.0 * jnp.pi) / 756.0
            - (2495.0 * eta * jnp.pi) / 378.0
            + (40.0 * integer_pow(eta, 2) * jnp.pi) / 27.0,
        ]
    )
    ampPNimag = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -24.0 * eta,
            (428.0 / 105.0) * jnp.pi,
            (14333.0 * eta) / 162.0 - (4066.0 * integer_pow(eta, 2)) / 945.0,
        ]
    )

    amplog = -428.0 / 105.0

    PNamplitude_coeffs = inspiral_coeffs | {
        "fac0": fac0,
        "ampN": ampN,
        "ampPNreal": ampPNreal,
        "ampPNimag": ampPNimag,
        "amplog": amplog,
        "ampInspC": jnp.zeros(3),
    }

    ampInspCP = jnp.array(
        [
            ansaetze.IMRPhenomT_Inspiral_Amp_CP1_22(eta, S, dchi, delta),
            ansaetze.IMRPhenomT_Inspiral_Amp_CP2_22(eta, S, dchi, delta),
            ansaetze.IMRPhenomT_Inspiral_Amp_CP3_22(eta, S, dchi, delta),
        ]
    )
    tinsppoints = jnp.array([-2000.0, -250.0, -150.0])

    theta = jax.lax.pow(-eta * tinsppoints / 5, -1.0 / 8)
    omega = jax.vmap(
        ansaetze.IMRPhenomTInspiralOmegaAnsatz22, in_axes=(0, None), out_axes=0
    )(theta, PNamplitude_coeffs)
    xx = jax.lax.pow(0.5 * omega, 2.0 / 3)
    ampoffset = jax.vmap(
        ansaetze.IMRPhenomTInspiralAmpAnsatzHM, in_axes=(0, None), out_axes=0
    )(xx, PNamplitude_coeffs)

    A = xx[:, None] ** (4 + 0.5 * jnp.arange(3))
    b = (ampInspCP - ampoffset) / (fac0 * xx)

    ampInspC = jnp.linalg.solve(A, b)

    return PNamplitude_coeffs | {"ampInspC": ampInspC}
