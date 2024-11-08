import numpy as np
from scipy.special import factorial, genlaguerre
from itertools import product

#


def HRF(Li, hO):
    return Li / hO  # Replace this with the actual function


# %% Franck-Condon Weighted Density Factor (downward transition)


def FCWD_v(State, const, hw, T):
    w_array = np.arange(0, State.vmlow)
    t_array = np.arange(0, State.vmhigh)
    # Create meshgrid for all combinations of T, hw, DG0, w, and t
    energy_offset = min(State.DG0-0.5)
    temp, hw_mat, dg0_mat, final_mat, initial_mat = np.meshgrid(T, hw, State.DG0, w_array, t_array)
    hrf = HRF(State.Li, State.hO)
    # hrf = hrf.numpy() if isinstance(hrf, at.tensor) else hrf

    lag = np.zeros((len(t_array), len(w_array)))

    for t, w in product(t_array, w_array):
        if w - t > -1:
                if t == 0:
                    lag[t, w] = 1
                else:
                    lag[t, w] = 0
        else:
            lag[t, w] = 0
    lagur = np.take(lag, initial_mat)
  

    # Calculate each component of the FCWD_n expression
    component1 = 1 / np.sqrt(4 * np.pi * State.Lo * const.kb * temp)
    component2 = np.exp(-hrf) * hrf ** (final_mat - initial_mat) * factorial(initial_mat) / factorial(final_mat)
    component3 = lagur**2
    component4 = np.exp(
        -((hw_mat - dg0_mat + State.Lo + (final_mat - initial_mat) * State.hO) ** 2)
        / (4 * State.Lo * const.kb * temp)
    )
    component5 = np.exp((-initial_mat * State.hO) / (np.array([const.kb]) * temp))
    component6 = np.exp((-dg0_mat+energy_offset) / (np.array([const.kb])* temp))
    FCWD_n = component1 * component2 * component3 * component4 * component5 * component6 
    FCWD_dw = np.sum(FCWD_n, axis=(3, 4))

    return FCWD_dw


# %% Franck-Condon Weighted Density Factor (upward transition)
def FCWD_n(State, const, hw, T):
    w_array = np.arange(0, State.vmhigh)
    t_array = np.arange(0, State.vmlow)
    # Create meshgrid for all combinations of T, hw, DG0, w, and t
    temp, hw_mat, dg0_mat, final_mat, initial_mat = np.meshgrid(T, hw, State.DG0, w_array, t_array)
    hrf = HRF(State.Li, State.hO)
    # hrf = hrf.numpy() if isinstance(hrf, at.tensor) else hrf

    lag = np.zeros((len(t_array), len(w_array)))

    for t, w in product(t_array, w_array):
        if w - t > -1:
            try:
                lag[t, w] = genlaguerre(t, w - t)(hrf)
            except:
                if t == 0:
                    lag[t, w] = 1
                else:
                    lag[t, w] = 0
        else:
            lag[t, w] = 0
    lagur = np.take(lag, initial_mat)
    # Calculate each component of the FCWD_n expression
    component1 = 1 / np.sqrt(4 * np.pi * State.Lo * const.kb * temp)
    component2 = np.exp(-hrf) * hrf ** (final_mat - initial_mat) * factorial(initial_mat) / factorial(final_mat)
    component3 = lagur**2
    component4 = np.exp(
        -((hw_mat - dg0_mat - State.Lo - (final_mat - initial_mat) * State.hO) ** 2)
        / (4 * np.abs(State.Lo) * const.kb * temp)
    )
    component5 = np.exp((-initial_mat * State.hO) / (const.kb * temp))
    FCWD_n = component1 * component2 * component3 * component4 * component5
    FCWD_uw = np.sum(FCWD_n, axis=(3, 4))


    return FCWD_uw


def E_Coup(State, const):
    M = tdm(State, const)
    V = (M * State.E) / np.sqrt(State.u**2 + 4 * M**2)
    return V


def tdm(State, const):
    M = np.sqrt((3 / 2) * const.h * const.hbar * State.fosc / (2 * np.pi * (State.E - State.hO) * const.m_e))
    return M


def Z_abs(State, xParam):
    dE = State.DG0[1] - State.DG0[0]  # the difference in value between each value of DG0 used for the integral

    # x * gauss + (1 - x) * exponential
    integral = (
        xParam["coeff"] * np.exp(-((State.DG0 - State.E) ** 2) / (2 * State.sigma**2))
        + (1 - xParam["coeff"]) * np.exp(-(State.DG0 - State.E) / State.sigma)
    ) * dE

    Zabs = np.sum(integral)

    return Zabs


def Z_rec(State, C, D, xParam):
    dE = State.DG0[1] - State.DG0[0]  # the difference in value between each value of DG0 used for the integral
    energy_offset = min(State.DG0-0.5)
    int = np.zeros((len(State.DG0), len(D.T)))
    for i in range(len(D.T)):
        for j in range(len(State.DG0)):
            int[j, i] = (
                (
                    xParam["coeff"] * np.exp(-((State.DG0[j] - State.E) ** 2) / (2 * State.sigma**2))
                    + (1 - xParam["coeff"]) * np.exp(-State.DG0[j] / (C.kb * D.T[i]))
                )
                * np.exp((-State.DG0[j]+energy_offset) / (C.kb * D.T[i]))
                * dE
            )

    Zrec = np.sum(int, axis=0)

    return Zrec
