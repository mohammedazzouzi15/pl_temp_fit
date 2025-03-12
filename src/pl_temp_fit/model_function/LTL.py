"""This module contains the LTL model functions and classes."""
import numpy as np

class State:
    """Class representing the state of the LTL model.

    Attributes:
        E (float): Energy.
        vmhigh (float): High voltage.
        vmlow (float): Low voltage.
        sigma (float): Sigma value.
        numbrstates (int): Number of states.
        off (float): Offset value.
        Li (float): Li value.
        Lo (float): Lo value.
        hO (float): hO value.
        fosc (float): Oscillator strength.
        dmus (float): Difference in static dipole moment.
        disorder_ext (float): Extension of the disorder Gaussian distribution in units of eV.
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
        """Initialize the state.

        Args:
            E (float): Energy.
            vmhigh (float): High voltage.
            vmlow (float): Low voltage.
            sigma (float): Sigma value.
            numbrstates (int): Number of states.
            off (float): Offset value.
            Li (float): Li value.
            Lo (float): Lo value.
            hO (float): hO value.
            fosc (float): Oscillator strength.
            dmus (float): Difference in static dipole moment.
            disorder_ext (float): Extension of the disorder Gaussian distribution in units of eV.
        """
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
        """Calculate the u value."""
        return self.dmus * 3.33e-30 / 1.6e-19

    def calculate_DG0(self):
        """Calculate the DG0 value."""
        return np.linspace(
            self.E - self.disorder_ext,
            self.E + self.disorder_ext,
            self.numbrstates,
        )

    def calculate_fosc(self):
        """Calculate the oscillator strength."""
        self.fosc = 10**self.log_fosc

    def update(self, **kwargs):
        """Update the state with new values.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.DG0 = self.calculate_DG0()
        self.u = self.calculate_u()
        self.calculate_fosc()
