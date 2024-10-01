import numpy as np
from itertools import product
from pl_temp_fit.model_function.UtilFunctions import tdm, Z_rec, Z_abs, E_Coup, FCWD_v, FCWD_n
from pl_temp_fit.model_function import LTL

# make the script useable by pytensors,
# the following lines are commented out because they are not used in the script


def krad(State, C, data, xParam):
    FCWD = FCWD_v(State, C, data.hw, data.T)
    M = tdm(State, C)
    Zrec = Z_rec(State, C, data, xParam)
    integral = np.zeros((len(data.T), len(data.hw), len(State.DG0)))
 
    if State.numbrstates == 1:
        dE = 0.01
        integral = (
            (4 / (3 * C.E0 * C.hbar**4))
            * ((data.hw.reshape(-1, 1, 1) / C.c) ** 3)
            * M**2
            * FCWD
            * dE
        )
       
    else:
        dE = State.DG0[1] - State.DG0[0]
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
    sum_integral = np.sum(integral, axis=2)
    State.kr_hw = sum_integral / Zrec
    State.M = M
    State.FCWD = FCWD
    dhw = data.hw[1] - data.hw[0]
    State.kr = np.sum(State.kr_hw * dhw, axis=0)
    return State

def kabs(State, C, data, xParam):
    FCWD = FCWD_n(State, C, data.hw, data.T)
    M = tdm(State, C)
    Zabs = Z_abs(State, xParam)
    
    if State.numbrstates == 1:
        dE = 0.01
        integral = (
            (data.n / C.c)
            * (1 / (6 * C.E0 * C.hbar))
            * M**2
            * FCWD
            * data.hw.reshape(-1, 1, 1)
            * dE
        )
    else:
        dE = State.DG0[1] - State.DG0[0]
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

    sum_integral = np.sum(integral, axis=2)
    State.ka_hw = sum_integral / Zabs

    return State


def knonrad(State, C, data, xParam):   
    FCWD = FCWD_v(State, C, [0], data.T)
    V = E_Coup(State, C)
    Zrec = Z_rec(State, C, data, xParam)
    kth = 1e14
    
    if State.numbrstates == 1:
        dE = 0.01
        integral = (
            ((2 * np.pi) / C.hbar)
            * V**2
            * FCWD
            * dE
        )
    else:
        dE = State.DG0[1] - State.DG0[0]
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
    sum_integral = np.sum(integral, axis=2) / Zrec
    State.knr = (sum_integral * kth) / (sum_integral + kth)

    return State
