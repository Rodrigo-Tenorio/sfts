"""
YOU SHOULD NOT BE LOOKING HERE UNLESS YOU KNOW WHAT YOU ARE DOING.
GO TO `iphenot.py` FOR THE PUBLIC API.

IMRPhenomT ansaetze adapted to work with `jax.jit` and `jax.vmap`.

See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTHM.c?ref_type=heads
and functions therein.
"""
import jax
import jax.numpy as jnp
from jax.lax import integer_pow

from .taylor_t3 import phenom_Taylor_T3


def IMRPhenomTInspiralPhaseAnsatz22(t, thetabar, coeffs):

    eta = coeffs["eta"]
    omg_n_PN = coeffs["omega_n_PN_coeffs"]
    omg_InspC = coeffs["omega_InspC_coeffs"]

    return (
        -(
            jax.lax.pow(5.0, -0.625)
            * jax.lax.pow(eta, -2)
            * jax.lax.pow(t, -1)
            * jax.lax.pow(thetabar, -7)
            * (
                3 * (-107.0 + 280.0 * omg_n_PN[4]) * jax.lax.pow(5.0, 0.75)
                + 321.0
                * jnp.log(thetabar * jax.lax.pow(5.0, 0.125))
                * jax.lax.pow(5.0, 0.75)
                + 420.0 * omg_n_PN[5] * thetabar * jax.lax.pow(5.0, 0.875)
                + 56.0 * (25 * omg_InspC[0] + 3 * eta * t) * jax.lax.pow(thetabar, 2)
                + 1050.0
                * omg_InspC[1]
                * jax.lax.pow(5.0, 0.125)
                * jax.lax.pow(thetabar, 3)
                + 280.0
                * (3 * omg_InspC[2] + eta * omg_n_PN[0] * t)
                * jax.lax.pow(5.0, 0.25)
                * jax.lax.pow(thetabar, 4)
                + 140.0
                * (5 * omg_InspC[3] + 3 * eta * omg_n_PN[1] * t)
                * jax.lax.pow(5.0, 0.375)
                * jax.lax.pow(thetabar, 5)
                + 120.0
                * (5 * omg_InspC[4] + 7 * eta * omg_n_PN[2] * t)
                * jax.lax.pow(5.0, 0.5)
                * jax.lax.pow(thetabar, 6)
                + 525.0
                * omg_InspC[5]
                * jax.lax.pow(5.0, 0.625)
                * jax.lax.pow(thetabar, 7)
                + 105.0
                * eta
                * omg_n_PN[3]
                * t
                * jnp.log(-t)
                * jax.lax.pow(5.0, 0.625)
                * jax.lax.pow(thetabar, 7)
            )
        )
        / 84.0
    )


def IMRPhenomTInspiralOmegaAnsatz22(theta, coeffs):
    taylort3 = phenom_Taylor_T3(theta, coeffs["omega_n_PN_coeffs"])
    fac = integer_pow(theta, 3) / 8
    out = (coeffs["omega_InspC_coeffs"] * theta ** jnp.arange(8, 14)).sum()
    return taylort3 + 2 * fac * out


def IMRPhenomTInspiralAmpAnsatzHM(x, coeffs):

    fac = coeffs["fac0"] * x
    x_PN = x ** (0.5 * jnp.arange(1, 8))
    x_C = x ** (4 + 0.5 * jnp.arange(3))

    ampreal = (
        coeffs["ampN"]
        + (coeffs["ampPNreal"] * x_PN).sum()
        + coeffs["amplog"] * jnp.log(16 * x) * x**3
    )

    ampimag = (coeffs["ampPNimag"] * x_PN).sum()

    amp = ampreal + (x_C * coeffs["ampInspC"]).sum() + 1j * ampimag

    return fac * amp


def IMRPhenomT_Inspiral_Freq_CP1_22(eta, S, dchi, delta):
    return (
        -0.014968864336704284
        * dchi
        * delta
        * (1 - 1.942061808318584 * eta)
        * integer_pow(eta, 2)
        + 0.0017312772309375462
        * dchi
        * delta
        * (1 - 0.07106994121956058 * eta)
        * S
        * integer_pow(eta, 2)
        + S
        * (
            0.0019208448318368731
            - 0.0013579968243452476 * eta
            - 0.0033501404728414627 * integer_pow(eta, 2)
            + 0.008914420175326192 * integer_pow(eta, 3)
        )
        + 6.687615165457298e-6 * integer_pow(dchi, 2) * integer_pow(eta, 3)
        + (
            0.02104073275966069
            + 717.1534194224539 * eta
            + 85.37320237350282 * integer_pow(eta, 2)
            + 12.789214868358362 * integer_pow(eta, 3)
            - 16.00243777208413 * integer_pow(eta, 4)
        )
        * integer_pow(1 + 32934.586638893634 * eta, -1)
        + (
            -8.306810248117731e-6
            + 0.00009918593182087119 * eta
            - 0.003805916669791129 * integer_pow(eta, 2)
            + 0.009854209286892323 * integer_pow(eta, 3)
        )
        * integer_pow(S, 2)
        + (
            -5.578836442449699e-6
            - 0.0030378960591856616 * eta
            + 0.03746366675135751 * integer_pow(eta, 2)
            - 0.10298471015315146 * integer_pow(eta, 3)
        )
        * integer_pow(S, 3)
        + (
            0.00004425141111368952
            - 0.0008702073302258368 * eta
            + 0.006538604805919268 * integer_pow(eta, 2)
            - 0.01578597166324495 * integer_pow(eta, 3)
        )
        * integer_pow(S, 4)
        + (
            -0.000019469656288570753
            + 0.002969863931498354 * eta
            - 0.03643271052162611 * integer_pow(eta, 2)
            + 0.09959495981802587 * integer_pow(eta, 3)
        )
        * integer_pow(S, 5)
        + (
            -0.000042037164406446896
            + 0.0007336074135429041 * eta
            - 0.005603356997202016 * integer_pow(eta, 2)
            + 0.013439843000090702 * integer_pow(eta, 3)
        )
        * integer_pow(S, 6)
    )


def IMRPhenomT_Inspiral_Freq_CP2_22(eta, S, dchi, delta):
    return (
        -0.04486391236129559
        * dchi
        * delta
        * (1 - 1.8997912248414794 * eta)
        * integer_pow(eta, 2)
        - 0.003531802135161727
        * dchi
        * delta
        * (1 - 8.001211450141325 * eta)
        * S
        * integer_pow(eta, 2)
        + S
        * (
            0.0061664395419698285
            - 0.0040934633081508905 * eta
            - 0.009180337242551828 * integer_pow(eta, 2)
            + 0.020338583755834694 * integer_pow(eta, 3)
        )
        + 0.00006524644306613066 * integer_pow(dchi, 2) * integer_pow(eta, 3)
        + integer_pow(1 - 3.2125452791404148 * eta, -1)
        * (
            0.03711511661217631
            - 0.10663782888636487 * eta
            - 0.09963406984414182 * integer_pow(eta, 2)
            + 0.6597367702009397 * integer_pow(eta, 3)
            - 2.777344875144891 * integer_pow(eta, 4)
            + 4.220674345359693 * integer_pow(eta, 5)
        )
        + (
            0.00044302547647888445
            + 0.000424246501303979 * eta
            - 0.01394093576260671 * integer_pow(eta, 2)
            + 0.02634851560709597 * integer_pow(eta, 3)
        )
        * integer_pow(S, 2)
        + (
            0.00011582043047950321
            - 0.008282652950117982 * eta
            + 0.08965067576998058 * integer_pow(eta, 2)
            - 0.23963885130463913 * integer_pow(eta, 3)
        )
        * integer_pow(S, 3)
        + (
            0.0006123158975881322
            - 0.007809160444435783 * eta
            + 0.028517174579539676 * integer_pow(eta, 2)
            - 0.03717957419042746 * integer_pow(eta, 3)
        )
        * integer_pow(S, 4)
        + (
            -0.0000885530893214531
            + 0.005939789043536808 * eta
            - 0.07106551435109858 * integer_pow(eta, 2)
            + 0.1891131957235774 * integer_pow(eta, 3)
        )
        * integer_pow(S, 5)
        + (
            -0.0005110853374341054
            + 0.0038762476596420855 * eta
            + 0.005094077179675256 * integer_pow(eta, 2)
            - 0.047971766995287136 * integer_pow(eta, 3)
        )
        * integer_pow(S, 6)
    )


def IMRPhenomT_Inspiral_Freq_CP3_22(eta, S, dchi, delta):
    return (
        -0.10196878573773932
        * dchi
        * delta
        * (1 - 1.8918584778973513 * eta)
        * integer_pow(eta, 2)
        - 0.018820536453940443
        * dchi
        * delta
        * (1 - 3.7307154599131183 * eta)
        * S
        * integer_pow(eta, 2)
        - 0.00013162098437956188 * integer_pow(dchi, 2) * integer_pow(eta, 3)
        + S
        * (
            0.0145572994468378
            - 0.0017482433991394227 * eta
            - 0.10299007619034371 * integer_pow(eta, 2)
            + 0.4581039376357615 * integer_pow(eta, 3)
            - 0.7123678787549022 * integer_pow(eta, 4)
        )
        + (
            0.05489007025458171
            + 5.852073438961151 * eta
            + 2.74597705533403 * integer_pow(eta, 2)
            + 4.834336623113389 * integer_pow(eta, 3)
            - 26.931994454691022 * integer_pow(eta, 4)
            + 57.67035368809743 * integer_pow(eta, 5)
        )
        * integer_pow(1 + 105.52132834236778 * eta, -1)
        + (
            0.003001211395915229
            + 0.0017929418998452987 * eta
            - 0.13776590125456148 * integer_pow(eta, 2)
            + 0.7471133710854526 * integer_pow(eta, 3)
            - 1.3620323111858437 * integer_pow(eta, 4)
        )
        * integer_pow(S, 2)
        + (
            0.001143282743686261
            - 0.05793457776296727 * eta
            + 0.7841331051705482 * integer_pow(eta, 2)
            - 3.4936244160305323 * integer_pow(eta, 3)
            + 4.802357041496856 * integer_pow(eta, 4)
        )
        * integer_pow(S, 3)
        + (
            0.0009168588840889624
            - 0.03261437094899735 * eta
            + 0.3472881896838799 * integer_pow(eta, 2)
            - 1.3634383958859384 * integer_pow(eta, 3)
            + 1.7313939586675267 * integer_pow(eta, 4)
        )
        * integer_pow(S, 4)
        + (
            -0.0002794014744432316
            + 0.055911057147527664 * eta
            - 0.8686311380514122 * integer_pow(eta, 2)
            + 4.096191294930781 * integer_pow(eta, 3)
            - 6.009676060669872 * integer_pow(eta, 4)
        )
        * integer_pow(S, 5)
        + (
            -0.0005046018052528331
            + 0.029804593053788925 * eta
            - 0.3792653361049425 * integer_pow(eta, 2)
            + 1.6366976231421981 * integer_pow(eta, 3)
            - 2.26904099961476 * integer_pow(eta, 4)
        )
        * integer_pow(S, 6)
    )


def IMRPhenomT_Inspiral_Freq_CP4_22(eta, S, dchi, delta):
    return (
        -0.1831889759662071
        * dchi
        * delta
        * (1 - 1.8484261527766557 * eta)
        * integer_pow(eta, 2)
        - 0.07586202965525136
        * dchi
        * delta
        * (1 - 3.2918162656371983 * eta)
        * S
        * integer_pow(eta, 2)
        + 0.0019259052728265817 * integer_pow(dchi, 2) * integer_pow(eta, 3)
        + S
        * (
            0.02685637375751212
            + 0.013341664908359861 * eta
            - 0.3057217933283597 * integer_pow(eta, 2)
            + 1.395763446325911 * integer_pow(eta, 3)
            - 2.2559396974665376 * integer_pow(eta, 4)
        )
        + (
            0.0725639467287476
            + 12.39400068457852 * eta
            + 12.907450928972402 * integer_pow(eta, 2)
            - 7.422660061864399 * integer_pow(eta, 3)
            + 66.32985901506036 * integer_pow(eta, 4)
            - 117.85875779454518 * integer_pow(eta, 5)
        )
        * integer_pow(1 + 168.63492460136445 * eta, -1)
        + (
            0.0087781653701194
            + 0.006944161553839352 * eta
            - 0.3301149078235105 * integer_pow(eta, 2)
            + 1.6835714783903248 * integer_pow(eta, 3)
            - 2.950404929598742 * integer_pow(eta, 4)
        )
        * integer_pow(S, 2)
        + (
            0.0037229746496019625
            - 0.17155338099487646 * eta
            + 2.5881802140836774 * integer_pow(eta, 2)
            - 13.14710199375518 * integer_pow(eta, 3)
            + 21.366803256010915 * integer_pow(eta, 4)
        )
        * integer_pow(S, 3)
        + (
            0.00278507305662002
            - 0.12475855143364532 * eta
            + 1.8640209516178643 * integer_pow(eta, 2)
            - 10.117078727717564 * integer_pow(eta, 3)
            + 17.94244821676711 * integer_pow(eta, 4)
        )
        * integer_pow(S, 4)
        + (
            0.0010273954584773936
            + 0.1713357629442166 * eta
            - 3.017249223460983 * integer_pow(eta, 2)
            + 15.855096360798678 * integer_pow(eta, 3)
            - 26.444621592311933 * integer_pow(eta, 4)
        )
        * integer_pow(S, 5)
        + (
            -0.00012207946532225968
            + 0.11709700788855186 * eta
            - 2.0950821618097026 * integer_pow(eta, 2)
            + 11.925324501640054 * integer_pow(eta, 3)
            - 21.683978511818076 * integer_pow(eta, 4)
        )
        * integer_pow(S, 6)
    )


def IMRPhenomT_Inspiral_Freq_CP5_22(eta, S, dchi, delta):
    return (
        -0.2508206617297265
        * dchi
        * delta
        * (1 - 1.861010982421798 * eta)
        * integer_pow(eta, 2)
        - 0.1392163711259171
        * dchi
        * delta
        * (1 - 3.2669366465555796 * eta)
        * S
        * integer_pow(eta, 2)
        + 0.0023126403170013045 * integer_pow(dchi, 2) * integer_pow(eta, 3)
        + S
        * (
            0.036750064163293766
            + 0.036904343404333906 * eta
            - 0.5238739410356437 * integer_pow(eta, 2)
            + 2.3292117112945223 * integer_pow(eta, 3)
            - 3.654184701923543 * integer_pow(eta, 4)
        )
        + (
            0.08373610487663233
            + 6.301736487754372 * eta
            + 9.03911386193751 * integer_pow(eta, 2)
            + 4.91153188278086 * integer_pow(eta, 3)
        )
        * integer_pow(1 + 72.64820846804257 * eta, -1)
        + (
            0.014963449678540705
            + 0.008354571522567225 * eta
            - 0.41723078020683 * integer_pow(eta, 2)
            + 2.2007932082378785 * integer_pow(eta, 3)
            - 4.245354787320365 * integer_pow(eta, 4)
        )
        * integer_pow(S, 2)
        + (
            0.005706180633326235
            - 0.15748500622007494 * eta
            + 2.3477109912232845 * integer_pow(eta, 2)
            - 11.413877195221694 * integer_pow(eta, 3)
            + 17.033120593116756 * integer_pow(eta, 4)
        )
        * integer_pow(S, 3)
        + (
            0.003890296981717687
            - 0.15985471334551038 * eta
            + 2.560312006077997 * integer_pow(eta, 2)
            - 14.400920672743332 * integer_pow(eta, 3)
            + 26.10406142567958 * integer_pow(eta, 4)
        )
        * integer_pow(S, 4)
        + (
            0.005305988847210204
            + 0.10869207132210629 * eta
            - 2.4201307115268875 * integer_pow(eta, 2)
            + 12.544899744864924 * integer_pow(eta, 3)
            - 19.550600837316903 * integer_pow(eta, 4)
        )
        * integer_pow(S, 5)
        + (
            0.002917248769788225
            + 0.11851143848720952 * eta
            - 2.6640023622893416 * integer_pow(eta, 2)
            + 15.993378498844761 * integer_pow(eta, 3)
            - 29.752144941054446 * integer_pow(eta, 4)
        )
        * integer_pow(S, 6)
    )


def IMRPhenomT_Inspiral_Amp_CP1_22(eta, S, dchi, delta):
    return (
        0.00006480771730217768 * eta * integer_pow(dchi, 2)
        - 0.3543965558027252
        * dchi
        * delta
        * (1 - 2.463526130684083 * eta)
        * integer_pow(eta, 3)
        + 0.01879295038873938
        * dchi
        * delta
        * (1 - 5.236796607517272 * eta)
        * S
        * integer_pow(eta, 3)
        + S
        * (
            0.1472653807120573 * eta
            - 1.9636752493349356 * integer_pow(eta, 2)
            + 14.177521724634461 * integer_pow(eta, 3)
            - 48.94620901701877 * integer_pow(eta, 4)
            + 63.83730899015984 * integer_pow(eta, 5)
        )
        + eta
        * (
            0.8493442097893826
            - 13.211067914003836 * eta
            + 311.99021467938235 * integer_pow(eta, 2)
            - 4731.025904601601 * integer_pow(eta, 3)
            + 44821.93042533854 * integer_pow(eta, 4)
            - 264474.1374080295 * integer_pow(eta, 5)
            + 943246.2317701122 * integer_pow(eta, 6)
            - 1.8588135904328802e6 * integer_pow(eta, 7)
            + 1.5524778581809246e6 * integer_pow(eta, 8)
        )
        + (
            0.04902976057622393 * eta
            - 1.0152511131279736 * integer_pow(eta, 2)
            + 8.286289152216145 * integer_pow(eta, 3)
            - 30.19775956110767 * integer_pow(eta, 4)
            + 40.670065442751955 * integer_pow(eta, 5)
        )
        * integer_pow(S, 2)
        + (
            0.04780630695082567 * eta
            - 1.2177827888317065 * integer_pow(eta, 2)
            + 11.505675146308567 * integer_pow(eta, 3)
            - 46.733420749352135 * integer_pow(eta, 4)
            + 68.40821782168776 * integer_pow(eta, 5)
        )
        * integer_pow(S, 3)
    )


def IMRPhenomT_Inspiral_Amp_CP2_22(eta, S, dchi, delta):

    return (
        0.000100027278976821 * eta * integer_pow(dchi, 2)
        - 0.7578403155712378
        * dchi
        * delta
        * (1 - 2.056456271350877 * eta)
        * integer_pow(eta, 3)
        - 0.14126282637778914
        * dchi
        * delta
        * (1 - 2.5840771007494916 * eta)
        * S
        * integer_pow(eta, 3)
        + S
        * (
            0.2331970217833686 * eta
            - 1.5473968380422929 * integer_pow(eta, 2)
            + 5.973401506474942 * integer_pow(eta, 3)
            - 9.110484789161045 * integer_pow(eta, 4)
        )
        + eta
        * (
            0.9904613241626621
            - 6.708006572605403 * eta
            + 127.40270095439482 * integer_pow(eta, 2)
            - 1723.355339710798 * integer_pow(eta, 3)
            + 15430.10086310527 * integer_pow(eta, 4)
            - 88744.26044058547 * integer_pow(eta, 5)
            + 313650.01696201024 * integer_pow(eta, 6)
            - 617887.8122937253 * integer_pow(eta, 7)
            + 518220.9267888211 * integer_pow(eta, 8)
        )
        + (
            0.08934817374146888 * eta
            - 0.8887847358339216 * integer_pow(eta, 2)
            + 3.7233864099350784 * integer_pow(eta, 3)
            - 5.814765403882651 * integer_pow(eta, 4)
        )
        * integer_pow(S, 2)
        + (
            0.04471990627820145 * eta
            - 0.642458648615624 * integer_pow(eta, 2)
            + 3.393481171493086 * integer_pow(eta, 3)
            - 6.092083983738554 * integer_pow(eta, 4)
        )
        * integer_pow(S, 3)
    )


def IMRPhenomT_Inspiral_Amp_CP3_22(eta, S, dchi, delta):
    return (
        0.0002459376633671657 * eta * integer_pow(dchi, 2)
        - 0.8794763631110696
        * dchi
        * delta
        * (1 - 2.0751630535350096 * eta)
        * integer_pow(eta, 3)
        - 0.3319387797134261
        * dchi
        * delta
        * (1 - 3.1838055629892184 * eta)
        * S
        * integer_pow(eta, 3)
        + S
        * (
            0.23505507416274007 * eta
            - 1.2449030421324767 * integer_pow(eta, 2)
            + 4.315803728759738 * integer_pow(eta, 3)
            - 6.384257606413192 * integer_pow(eta, 4)
        )
        + eta
        * (
            1.0208762064809185
            - 3.3799457394243957 * eta
            + 16.242639717123314 * integer_pow(eta, 2)
            + 299.2297416582362 * integer_pow(eta, 3)
            - 5913.920743907752 * integer_pow(eta, 4)
            + 46388.231537995445 * integer_pow(eta, 5)
            - 192261.0498470111 * integer_pow(eta, 6)
            + 413750.14250475995 * integer_pow(eta, 7)
            - 364403.84935539874 * integer_pow(eta, 8)
        )
        + (
            0.09630827896641526 * eta
            - 0.7915321134872877 * integer_pow(eta, 2)
            + 2.86907420250287 * integer_pow(eta, 3)
            - 4.038995403653199 * integer_pow(eta, 4)
        )
        * integer_pow(S, 2)
        + (
            0.07395420485618898 * eta
            - 1.0289224187583748 * integer_pow(eta, 2)
            + 5.275845823734598 * integer_pow(eta, 3)
            - 9.206158044409037 * integer_pow(eta, 4)
        )
        * integer_pow(S, 3)
    )
