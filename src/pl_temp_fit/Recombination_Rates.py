import numpy as np
from itertools import product
from pl_temp_fit.Util_functions import tdm, Z_rec, Z_abs, E_Coup, FCWD_v, FCWD_n

# make the script useable by pytensors,
# the following lines are commented out because they are not used in the script


def krad(State, C, data, xParam):
    dE = State.DG0[1] - State.DG0[0]
    FCWD = FCWD_v(State, C, data.hw, data.T)
    M = tdm(State, C)
    Zrec = Z_rec(State, C, data, xParam)
    integral = np.zeros((len(data.T), len(data.hw), len(State.DG0)))
 
    integral = (
        (4 / (3 * C.E0 * C.hbar**4))
        * ((data.hw.reshape(-1, 1, 1) / C.c) ** 3)
        * M**2
        * FCWD
        * (
            xParam["coeff"] * np.exp(-((State.DG0.reshape(1, 1, -1) - State.E) ** 2) / (2 * State.sigma**2))
            + (1 - xParam["coeff"]) * np.exp(-(State.DG0.reshape(1, 1, -1) - State.E) / State.sigma)
        )
        * dE
    )
    if type(integral) is np.ndarray:
        sum_integral = np.sum(integral, axis=2)
    else:
        from pymc import math
        sum_integral = math.sum(integral, axis=2)
    State.kr_hw = sum_integral / Zrec
    State.M = M
    State.FCWD = FCWD
    dhw = data.hw[1] - data.hw[0]
    #State.kr = np.sum(State.kr_hw * dhw, axis=0)
    if type(State.kr_hw) is np.ndarray:
        State.kr = np.sum(State.kr_hw * dhw, axis=0)
    else:
        from pymc import math
        State.kr = math.sum(State.kr_hw * dhw, axis=0)
    return State


def kabs(State, C, data, xParam):
    dE = State.DG0[1] - State.DG0[0]
    FCWD = FCWD_n(State, C, data.hw, data.T)
    M = tdm(State, C)
    Zabs = Z_abs(State, xParam)
    """
    integral = np.zeros((len(data.T), len(data.hw), len(State.DG0)))

    for i, j, k in product(range(len(data.T)), range(len(data.hw)), range(len(State.DG0))):
        x = (
            (data.n / C.c)
            * (1 / (6 * C.E0 * C.hbar))
            * M**2
            * FCWD[j, i, k]
            * data.hw[j]
            * (
                xParam["coeff"] * np.exp(-((State.DG0[k] - State.E) ** 2) / (2 * State.sigma**2))
                + (1 - xParam["coeff"]) * np.exp(-(State.DG0[k] - State.E) / State.sigma)
            )
            * dE
        )
        integral[i, j, k] = x
    """
    integral = (
        (data.n / C.c)
        * (1 / (6 * C.E0 * C.hbar))
        * M**2
        * FCWD
        * data.hw.reshape(-1, 1, 1)
        * (
            xParam["coeff"] * np.exp(-((State.DG0.reshape(1, 1, -1) - State.E) ** 2) / (2 * State.sigma**2))
            + (1 - xParam["coeff"]) * np.exp(-(State.DG0.reshape(1, 1, -1) - State.E) / State.sigma)
        )
        * dE
    )
    #sum_integral = np.sum(integral, axis=2)
    if type(integral) is np.ndarray:
        sum_integral = np.sum(integral, axis=2)
    else:
        from pymc import math
        sum_integral = math.sum(integral, axis=2)
    State.ka_hw = sum_integral / Zabs

    return State


def knonrad(State, C, data, xParam):
    dE = State.DG0[1] - State.DG0[0]
    FCWD = FCWD_v(State, C, [0], data.T)
    V = E_Coup(State, C)
    Zrec = Z_rec(State, C, data, xParam)
    kth = 1e14
    """
    integral = np.zeros((len(data.T), len(State.DG0)))

    for i, k in product(range(len(data.T)), range(len(State.DG0))):
        integral[i, k] = (
            ((2 * np.pi) / C.hbar)
            * V**2
            * FCWD[0, i, k]
            * (
                xParam["coeff"] * np.exp(-((State.DG0[k] - State.E) ** 2) / (2 * State.sigma**2))
                + (1 - xParam["coeff"]) * np.exp(-(State.DG0[k] - State.E) / State.sigma)
            )
            * dE
        )
    """
    integral = (
            ((2 * np.pi) / C.hbar)
            * V**2
            * FCWD
            * (
                xParam["coeff"] * np.exp(-((State.DG0.reshape(1,-1) - State.E) ** 2) / (2 * State.sigma**2))
                + (1 - xParam["coeff"]) * np.exp(-(State.DG0.reshape(1,-1) - State.E) / State.sigma)
            )
            * dE
        )
    State.V = V
    State.Zrec = Zrec
    #sum_integral = np.sum(integral, axis=2) / Zrec
    if type(integral) is np.ndarray:
        sum_integral = np.sum(integral, axis=2) / Zrec
    else:
        from pymc import math
        sum_integral = math.sum(integral, axis=2) / Zrec
    State.knr = (sum_integral * kth) / (sum_integral + kth)

    return State
