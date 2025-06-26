"""Functions to simulate the low temperature luminescence spectra of organic semiconductors and the definition of the different classes used in the simulation.

the classes considered are the following:
- Constants: contains the constants used in the simulation
- Data: contains the data used in the simulation
- State: contains the state of the system
- DataParams: contains the parameters of the data
- LightSource: contains the light source used in the simulation

the functions considered are the following:
- ltlcalc: calculates the low temperature luminescence spectra of organic semiconductors
"""

import numpy as np

from pl_temp_fit.model_function.abs_events import calculate_alpha
from pl_temp_fit.model_function.ElEvents import el_gen
from pl_temp_fit.model_function.PlEvents import SUM, Emi, Gen
from pl_temp_fit.model_function.RecombinationRates import kabs, knonrad, krad


def ltlcalc(data):
    """Calculate the low temperature luminescence spectra of organic semiconductors.

    Args:
    ----
        data (Data): The data used in the simulation.

    Returns:
    -------
        data (Data): the pl data class with the calculated properties of the system and the different states


    """
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
        data.EX, data.CT = el_gen(data.EX, data.CT, data.D)
    else:
        raise ValueError("Luminecence_exp must be either PL or EL")
    # Kinetic movements leading to changes in the Density of States
    data.EX, data.CT = SUM(data.EX, data.CT, data.c, data.D)
    # Emission of Photons
    data.D = Emi(data.EX, data.CT, data.D)
    # Calculate the absorption of the system
    data.D.alpha = calculate_alpha(data.EX, data.CT, data.D)
    return data


class Constants:
    """Constants used in the simulation.

    Attributes
    ----------
    - c (float): Speed of light in vacuum
    - q (float): Charge of an electron
    - m_e (float): Mass of an electron
    - kb (float): Boltzmann constant(eV/K)
    - E0 (float): Vacuum permittivity (eV * Angstroms / (elementary charge)^2)
    - h (float): Planck's constant
    - hbar (float): Reduced Planck's constant

    """

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
    """Data used in the simulation.

    Attributes
    ----------
    - EX (State): The state of the system
    - CT (State): The state of the system
    - D (DataParams): The parameters of the data
    - I0 (LightSource): The light source used in the simulation
    - c (Constants): The constants used in the simulation
    - xParam (dict): The parameters of the simulation

    """

    def __init__(self):
        """Initialize the data used in the simulation."""
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
        self.voltage_results = {}

    def update(self, **kwargs: dict):
        """Update the data used in the simulation."""
        self.EX.update(**kwargs["EX"])
        self.CT.update(**kwargs["CT"])
        self.D.update(**kwargs["D"])

    def get_delta_voc_nr(self):
        raditaive_decay = self.CT.kr + self.EX.kr
        non_radiative_decay = self.CT.knr + self.EX.knr
        self.voltage_results["internal_quantum_efficiency"] = (
            raditaive_decay / (raditaive_decay + non_radiative_decay)
        )
        pe = 0.21  # Here we assume that theemission probability that is  ration of J0rad  by the integrated radiative recombination is 0.21
        self.voltage_results["external_quantum_efficiency"] = 1 / (
            (
                1
                + (pe - 1)
                * self.voltage_results["internal_quantum_efficiency"]
            )
            / self.voltage_results["internal_quantum_efficiency"]
            / pe
        )
        self.voltage_results["delta_voc_nr"] = (
            self.c.kb
            * self.D.T
            * np.log(1 / self.voltage_results["external_quantum_efficiency"])
        )


class State:
    """The class of the excited state of the system.

    Attributes
    ----------
    - E (float): The mean energy of the state
    - knr (float): The non-radiative recombination rate
    - kr (float): The radiative recombination rate
    - ka_hw (float): The absorption rate
    - vmhigh : number of high vibrational states to consider
    - vmlow : number of low vibrational states to consider
    - sigma : The standard deviation of the state energy following a Gaussian distribution
    - numbrstates : The number of states to consider
    - Gen : The generation rate of the state
    - Sum : The sum of the state
    - off : The state is off
    - DG0 : The gibbs free energy of the states
    - hO : The vibronic mode energy
    - Li : The high frequency reorganization energy
    - Lo : The low frequency reorganization energy
    - fosc : The oscillator strength
    - dmus : The diufference in static dipole moment
    - disorder_ext : The extension o fthe disorder gaussian distribution now defined in units of eV

    """

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
        disorder_ext=0.1,
    ):
        self.E = E  # mean energy of the state
        self.knr = 0
        self.kr = 0
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
        self.disorder_ext = disorder_ext

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
            self.E - self.disorder_ext,
            self.E + self.disorder_ext,
            self.numbrstates,
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
        self.nie = 1.5
        self.Excitondesnity = 1 / np.power(5e-10, 3)  # in unit m^-3

    def calculate_kEXCT(self):
        self.kEXCT = 10**self.log_kEXCT

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.calculate_kEXCT()


class LightSource:
    def __init__(self):
        self.Laser_hw = 1.7  # Example value, replace with your actual data
        self.Laser_B = 0.1  # Example value, replace with your actual data
