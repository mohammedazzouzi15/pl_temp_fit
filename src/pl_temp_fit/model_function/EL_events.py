import numpy as np
from scipy.integrate import trapz


def EL_gen(EX, CT, D):
    if CT.off == 1:
        Gen_EX = np.zeros(len(D.T))
        EX.Gen = np.zeros(len(D.T))
        Gen_EX = 1 # trapz(EX.ka_hw * np.exp(-((D.hw.reshape(-1,1) - I0.Laser_hw) ** 2) / I0.Laser_B**2), axis=0)
        EX.Gen = 1e25 * Gen_EX / Gen_EX
        CT.Gen = "state is off"
    else:
        Gen_EX = np.zeros(len(D.T))
        Gen_CT = np.zeros(len(D.T))
        EX.Gen = np.zeros(len(D.T))
        CT.Gen = np.zeros(len(D.T))
        Gen_EX = 0
        Gen_CT = 1 # all generation goes through the CT
        EX.Gen = 1e25 * Gen_EX / (Gen_EX+Gen_CT)
        CT.Gen = 1e25 * Gen_CT / (Gen_EX+Gen_CT)

    return EX, CT

