""" Class to fit the PL data with a model that includes the absorption peak and lifetime parameters. """
from pathlib import Path
import numpy as np
from pl_temp_fit.data_generators.SpectralDataGeneration import SpectralDataGeneration

class PLAbsAndLifetime(SpectralDataGeneration):
    def __init__(self, temperature_list, hws_list):
        super().__init__(temperature_list, hws_list)
        self.dtypes = [
            ("log_likelihood", float),
            ("Chi square", float),
            ("Ex_kr", float),
            ("Ex_knr", float),
            ("max_abs_pos", float),
        ]
        self.max_abs_pos_exp = 1.5 # in eV
        self.error_in_max_abs_pos = 0.01 # in eV
        self.lifetime_exp_high_temp = 1e-9 # in seconds
        self.error_in_lifetime_high_temp = 1e-10 # in seconds

    def log_probability(self, theta, exp_data, inv_co_var_mat_pl):
        """this is an example of a log probability function for the model."""
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return -np.inf, None, None, None, None, None # do not forget to return the correct number of outputs
        log_like, chi_squared, data = self.log_likelihood(
            theta,
            exp_data,
            inv_co_var_mat_pl,
        )
        log_prob = lp + log_like[0][0]
        log_prob += -(data.D.hw[data.D.alpha[:, -1].argmax()]   - self.max_abs_pos_exp)**2 / self.error_in_max_abs_pos**2/2
        exciton_lifetime = 1/ (data.EX.knr[0][-1] + data.EX.kr[-1])
        log_prob += -(exciton_lifetime - self.lifetime_exp_high_temp)**2 / self.error_in_lifetime_high_temp**2/2
        if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
            return -np.inf, None, None, None, None,None# do not forget to return the correct number of outputs
        return (
            log_prob,
            log_like[0],
            chi_squared[0][0],
            data.EX.knr[0][-1],
            data.EX.kr[-1],
            data.D.hw[data.D.alpha[:, -1].argmax()]  
        )# do not forget to return the correct number of outputs