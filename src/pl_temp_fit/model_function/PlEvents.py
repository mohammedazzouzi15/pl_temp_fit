import numpy as np
from scipy.integrate import trapezoid


def Gen(EX, CT, I0, D):
    if CT.off == 1:
        Gen_EX = np.zeros(len(D.T))
        EX.Gen = np.zeros(len(D.T))

        Gen_EX = 1
        EX.Gen = 1e25 * Gen_EX / Gen_EX
        CT.Gen = "state is off"

    else:
        Gen_EX = np.zeros(len(D.T))
        Gen_CT = np.zeros(len(D.T))
        EX.Gen = np.zeros(len(D.T))
        CT.Gen = np.zeros(len(D.T))

        Gen_EX = trapezoid(
            EX.ka_hw
            * np.exp(
                -((D.hw.reshape(-1, 1) - I0.Laser_hw) ** 2) / I0.Laser_B**2
            ),
            axis=0,
        )
        Gen_CT = trapezoid(
            CT.ka_hw
            * np.exp(
                -((D.hw.reshape(-1, 1) - I0.Laser_hw) ** 2) / I0.Laser_B**2
            ),
            axis=0,
        )
        EX.Gen = 1e25 * Gen_EX / (Gen_EX + Gen_CT)
        CT.Gen = 1e25 * Gen_CT / (Gen_EX + Gen_CT)

    return EX, CT


def Emi(EX, CT, D):
    if CT.off == 1:
        D.kr_hw = EX.kr_hw * EX.Sum
    else:
        D.kr_hw = EX.kr_hw * EX.Sum + CT.kr_hw * CT.Sum

    return D


def SUM(EX, CT, C, D):
    if CT.off == 1:
        EXgen = EX.Gen
        EX.Sum = np.zeros(len(D.T))
        krecEX = EX.knr + EX.kr
        ni = np.sqrt(D.DoS**2) * np.exp(-EX.E / (2 * C.kb * D.T))
        EX0 = ni * ni
        EX.Sum = (EXgen + krecEX * EX0) / krecEX
        CT.Sum = "state is off"

    else:
        EX.Sum = np.zeros(len(D.T))
        CT.Sum = np.zeros(len(D.T))
        kEXCT = D.kEXCT
        kCTEX = kEXCT * np.exp(-(EX.E - CT.E) / (C.kb * D.T)) / D.RCTE
        EXgen = EX.Gen
        CTgen = CT.Gen
        EX.Sum = np.zeros(len(D.T))
        CT.Sum = np.zeros(len(D.T))
        krecEX = EX.knr + EX.kr
        krecCT = CT.knr + CT.kr
        ni = np.sqrt(D.DoS**2) * np.exp(-CT.E / (2 * C.kb * D.T))
        CT0 = ni * ni
        EX0 = CT0 * np.exp(-(EX.E - CT.E) / C.kb * D.T) / D.RCTE
        EX.Sum = (
            EXgen
            + CTgen
            + krecEX * EX0
            + krecCT * CT0
            + krecCT / kCTEX * EXgen
            + krecCT * krecEX / kCTEX * EX0
        ) / (krecEX + krecCT * kEXCT / kCTEX + krecCT * krecEX / kCTEX)
        CT.Sum = (
            CTgen
            + EXgen
            + krecCT * CT0
            + krecEX * EX0
            + krecEX / kEXCT * CTgen
            + krecEX * krecCT / kEXCT * CT0
        ) / (krecCT + krecEX * kCTEX / kEXCT + krecEX * krecCT / kEXCT)
    return EX, CT
