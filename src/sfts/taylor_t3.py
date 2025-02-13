"""
YOU SHOULD NOT BE LOOKING HERE UNLESS YOU KNOW WHAT YOU ARE DOING.
GO TO `iphenot.py` FOR THE PUBLIC API.

IMRPhenomT Taylor T3 apporoximant adapted to work with `jax.jit` and `jax.vmap`.

See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTHM.c?ref_type=heads
and functions therein.
"""
from jax.lax import integer_pow
import jax.numpy as jnp

def phenom_Taylor_T3(theta, omega_n_PN_coeffs):
    # IMRPhenomTTaylorT3
    fac = integer_pow(theta, 3) / 8
    logterm = (107 * jnp.log(theta)) / 280.0
    out = (
        1
        + (omega_n_PN_coeffs * (theta ** jnp.arange(2, 8))).sum()
        + logterm * integer_pow(theta, 6)
    )
    return 2 * fac * out


def omega_n_PN(eta, delta, chi1L, chi2L):
    # returns (omega1PN, omega1halfPN, ..., omega3halfPN)
    return jnp.array(
        [
            0.27641369047619047 + (11 * eta) / 32.0,
            (-19 * (chi1L + chi2L) * eta) / 80.0
            + (-113 * (chi2L * (-1 + delta) - chi1L * (1 + delta)) - 96 * jnp.pi)
            / 320.0,
            (
                1855099.0
                + 1714608.0 * chi2L * chi2L * (-1 + delta)
                - 1714608.0 * chi1L * chi1L * (1 + delta)
            )
            / 1.4450688e7
            + (
                (
                    56975.0
                    + 61236.0 * chi1L * chi1L
                    - 119448.0 * chi1L * chi2L
                    + 61236.0 * chi2L * chi2L
                )
                * eta
            )
            / 258048.0
            + (371 * eta * eta) / 2048.0,
            (-17 * (chi1L + chi2L) * eta * eta) / 128.0
            + (-146597 * (chi2L * (-1 + delta) - chi1L * (1 + delta)) - 46374 * jnp.pi)
            / 129024.0
            + (
                eta
                * (
                    -2 * (chi1L * (1213 - 63 * delta) + chi2L * (1213 + 63 * delta))
                    + 117 * jnp.pi
                )
            )
            / 2304.0,
            -2.499258364444952
            - (16928263.0 * chi1L * chi1L) / 1.376256e8
            - (16928263.0 * chi2L * chi2L) / 1.376256e8
            - (16928263.0 * chi1L * chi1L * delta) / 1.376256e8
            + (16928263.0 * chi2L * chi2L * delta) / 1.376256e8
            + (
                (
                    -2318475.0
                    + 18767224.0 * chi1L * chi1L
                    - 54663952.0 * chi1L * chi2L
                    + 18767224.0 * chi2L * chi2L
                )
                * eta
                * eta
            )
            / 1.376256e8
            + (235925.0 * eta * eta * eta) / 1.769472e6
            + (107.0 * jnp.euler_gamma) / 280.0
            - (6127.0 * chi1L * jnp.pi) / 12800.0
            - (6127.0 * chi2L * jnp.pi) / 12800.0
            - (6127.0 * chi1L * delta * jnp.pi) / 12800.0
            + (6127.0 * chi2L * delta * jnp.pi) / 12800.0
            + (53.0 * jnp.pi * jnp.pi) / 200.0
            + (
                eta
                * (
                    632550449425.0
                    + 35200873512.0 * chi1L * chi1L
                    - 28527282000.0 * chi1L * chi2L
                    + 9605339856.0 * chi1L * chi1L * delta
                    - 1512.0 * chi2L * chi2L * (-23281001.0 + 6352738.0 * delta)
                    + 34172264448.0 * (chi1L + chi2L) * jnp.pi
                    - 22912243200.0 * jnp.pi * jnp.pi
                )
            )
            / 1.040449536e11
            + (107 * jnp.log(2)) / 280.0,
            (-12029.0 * (chi1L + chi2L) * eta * eta * eta) / 92160.0
            + (
                eta
                * eta
                * (
                    507654.0 * chi1L * chi2L * chi2L
                    - 838782.0 * chi2L * chi2L * chi2L
                    + chi2L * (-840149.0 + 507654.0 * chi1L * chi1L - 870576.0 * delta)
                    + chi1L * (-840149.0 - 838782.0 * chi1L * chi1L + 870576.0 * delta)
                    + 1701228.0 * jnp.pi
                )
            )
            / 1.548288e7
            + (
                eta
                * (
                    218532006.0 * chi1L * chi2L * chi2L * (-1 + delta)
                    - 1134.0 * chi2L * chi2L * chi2L * (-206917.0 + 71931.0 * delta)
                    - chi2L
                    * (
                        1496368361.0
                        - 429508815.0 * delta
                        + 218532006.0 * chi1L * chi1L * (1 + delta)
                    )
                    + chi1L
                    * (
                        -1496368361.0
                        - 429508815.0 * delta
                        + 1134.0 * chi1L * chi1L * (206917.0 + 71931.0 * delta)
                    )
                    - 144.0
                    * (
                        488825.0
                        + 923076.0 * chi1L * chi1L
                        - 1782648.0 * chi1L * chi2L
                        + 923076.0 * chi2L * chi2L
                    )
                    * jnp.pi
                )
            )
            / 1.8579456e8
            + (
                -6579635551.0 * chi2L * (-1 + delta)
                + 535759434.0 * chi2L * chi2L * chi2L * (-1 + delta)
                - chi1L * (-6579635551.0 + 535759434.0 * chi1L * chi1L) * (1 + delta)
                + (
                    -565550067.0
                    - 465230304.0 * chi2L * chi2L * (-1 + delta)
                    + 465230304.0 * chi1L * chi1L * (1 + delta)
                )
                * jnp.pi
            )
            / 1.30056192e9,
        ]
    )


def t0(eta, S, dchi, delta):
    return integer_pow(eta, -1) * (
        (-20.74399646637014 - 106.27711276502542 * eta)
        * integer_pow(1 + 0.6516016033332481 * eta, -1)
        + 0.0012450290074562259
        * dchi
        * delta
        * (1 - 4.701633367918768e6 * eta)
        * integer_pow(eta, 2)
        - 111.5049997379579
        * dchi
        * delta
        * (1 + 19.95458485773613 * eta)
        * S
        * integer_pow(eta, 2)
        + 1204.6829118499857
        * (1 - 4.025474056585855 * eta)
        * integer_pow(dchi, 2)
        * integer_pow(eta, 3)
        + S
        * (
            338.7318821277009
            - 1553.5891860091408 * eta
            + 19614.263378999745 * integer_pow(eta, 2)
            - 156449.78737303324 * integer_pow(eta, 3)
            + 577363.3090369126 * integer_pow(eta, 4)
            - 802867.433363341 * integer_pow(eta, 5)
        )
        + (
            -55.75053935847546
            - 290.36341163610575 * eta
            + 7873.7667183299345 * integer_pow(eta, 2)
            - 43585.59040070178 * integer_pow(eta, 3)
            + 87229.84668746481 * integer_pow(eta, 4)
            - 32469.263449695136 * integer_pow(eta, 5)
        )
        * integer_pow(S, 2)
        + (
            -102.8269343111326
            + 5121.845705262981 * eta
            - 93026.46878769135 * integer_pow(eta, 2)
            + 650989.6793529999 * integer_pow(eta, 3)
            - 1.8846061037110784e6 * integer_pow(eta, 4)
            + 1.861602620702142e6 * integer_pow(eta, 5)
        )
        * integer_pow(S, 3)
        + (
            -7.294950933078567
            + 314.24955197427136 * eta
            - 3751.8509582195657 * integer_pow(eta, 2)
            + 21205.339564205595 * integer_pow(eta, 3)
            - 46448.94771114493 * integer_pow(eta, 4)
            + 20310.512558558552 * integer_pow(eta, 5)
        )
        * integer_pow(S, 4)
        + (
            97.22312282683716
            - 4556.60375328623 * eta
            + 76308.73046927384 * integer_pow(eta, 2)
            - 468784.4188333802 * integer_pow(eta, 3)
            + 998692.0246600509 * integer_pow(eta, 4)
            - 322905.9042578296 * integer_pow(eta, 5)
        )
        * integer_pow(S, 5)
    )
