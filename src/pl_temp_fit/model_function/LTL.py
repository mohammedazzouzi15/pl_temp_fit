# function to simulate the low temperature luminescence spectra of organic semiconductors

import numpy as np

from pl_temp_fit.model_function.PL_Events import Gen, Emi, SUM
from pl_temp_fit.model_function.Recombination_Rates import krad, kabs, knonrad
from pl_temp_fit.model_function.EL_events import EL_gen


def LTLCalc(data):
    # Calculate the LTPL data
    data.EX = krad(data.EX, data.c, data.D, data.xParam)
    data.EX = knonrad(data.EX, data.c, data.D, data.xParam)
    data.EX = kabs(data.EX, data.c, data.D, data.xParam)
    if data.CT.off == 0:
        data.CT = krad(data.CT, data.c, data.D, data.xParam)
        data.CT = knonrad(data.CT, data.c, data.D, data.xParam)
        data.CT = kabs(data.CT, data.c, data.D, data.xParam)
    # Absorption of Photons and Generation of Excitons
    if data.D.Luminecence_exp == "PL":
        data.EX, data.CT = Gen(data.EX, data.CT, data.I0, data.D)
    elif data.D.Luminecence_exp == "EL":
        data.EX, data.CT = EL_gen(data.EX, data.CT, data.D)
    else:
        raise ValueError("Luminecence_exp must be either PL or EL")
    # Kinetic movements leading to changes in the Density of States
    data.EX, data.CT = SUM(data.EX, data.CT, data.c, data.D)
    # Emission of Photons
    data.D = Emi(data.EX, data.CT, data.D)
    return data


class Constants:
    c = 3e8  # Speed of light in vacuum
    q = 1.6e-19  # % Charge of an electron
    m_e = 9.1e-31  # Mass of an electron
    kb = np.array([8.6173303e-5])  # Boltzmann constant(eV/K)
    E0 = (
        4 * np.pi * 8.85e-12 * 6.242e18
    )  # Vacuum permittivity (eV * Angstroms / (elementary charge)^2)
    h = 6.62e-34  # Planck's constant
    hbar = h / q / (2 * np.pi)  # Reduced Planck's constant


class Data:
    def __init__(self):
        self.EX = State(
            E=1.4,
            vmhigh=2,
            vmlow=15,
            sigma=0.02,
            numbrstates=20,
            off=0,
            Li=0.1,  # Fix: Changed 'LI' to 'Li'
            Lo=0.1,
            hO=0.15,
            fosc=5,
        )
        self.CT = State(
            E=1.1,
            vmhigh=2,
            vmlow=15,
            sigma=0.01,
            numbrstates=20,
            off=1,
            Li=0.1,  # Fix: Changed 'LI' to 'Li'
            Lo=0.1,
            hO=0.15,
            fosc=5,
        )
        self.D = DataParams()
        self.I0 = LightSource()
        self.c = Constants()
        self.xParam = {"coeff": 1}
    def update(self, **kwargs):
        #self.__dict__.update(kwargs)
        self.EX.update(**kwargs["EX"])
        self.CT.update(**kwargs["CT"])
        self.D.update(**kwargs["D"])


class State:
    def __init__(
        self,
        E,
        vmhigh,
        vmlow,
        sigma,
        numbrstates,
        off,
        Li=0.1,
        Lo=0.1,
        hO=0.15,
        fosc=0.5,
        dmus=3.0,
    ):
        self.E = E  # mean energy of the state
        self.knr = None
        self.kr = None
        self.ka_hw = None
        self.vmhigh = (
            vmhigh  # number of vibrational states above the initial state
        )
        self.vmlow = (
            vmlow  # number of vibrational states below the final state
        )
        self.sigma = sigma
        self.numbrstates = numbrstates
        self.Gen = None
        self.Sum = None
        self.off = off
        self.DG0 = self.calculate_DG0()
        self.hO = hO  # vibronic mode energy
        self.Li = Li  # high frequency reorganization energy
        self.Lo = Lo  # low frequency reorganization energy
        self.fosc = fosc
        self.dmus = dmus
        self.u = self.calculate_u()
        self.log_fosc = np.log10(fosc)

    def calculate_u(self):
        return self.dmus * 3.33e-30 / 1.6e-19

    def calculate_DG0(self):
        return np.linspace(
            self.E - 5 * self.sigma, self.E + 5 * self.sigma, self.numbrstates
        )
    
    def calculate_fosc(self):
        self.fosc = 10**self.log_fosc

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.DG0 = self.calculate_DG0()
        self.u = self.calculate_u()
        self.calculate_fosc()


class DataParams:
    def __init__(self):
        self.T = np.linspace(
            90, 300, 5
        )  # Example value, replace with your actual data
        self.DoS = 1e5  # Example value, replace with your actual data
        self.kEXCT = 1e11  # Example value, replace with your actual data
        self.RCTE = 2.0  # Example value, replace with your actual data
        self.hw = np.arange(0, 5, 0.01)
        self.n = 1.0
        self.Luminecence_exp = "PL"  # 'PL' or 'EL
        self.log_kEXCT = 11
    def calculate_kEXCT(self):
        self.kEXCT= 10 ** self.log_kEXCT

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.calculate_kEXCT()


class LightSource:
    def __init__(self):
        self.Laser_hw = 1.7  # Example value, replace with your actual data
        self.Laser_B = 0.1  # Example value, replace with your actual data