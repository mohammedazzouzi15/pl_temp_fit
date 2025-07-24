"""Class to fit the PL data with a model that includes the absorption peak and lifetime parameters for all temps HY 08/10/2024."""

import logging

import numpy as np

from pl_temp_fit.data_generators.SpectralDataGeneration import (
    SpectralDataGeneration,
)


class PLAbsandAlllifetime(SpectralDataGeneration):
    def __init__(self, temperature_list, hws_list):
        super().__init__(temperature_list, hws_list)
        self.max_abs_pos_exp = 1.5  # in eV
        self.error_in_max_abs_pos = 0.01  # in eV
        self.relative_error_lifetime = 0.05  # 5% relative error
        self.temperature_lifetimes_exp = {}  # in seconds
        self.temperature_lifetimes_exp = {}  # in seconds
        self.check_available_lifetime()
        self.dtypes = self.generate_dtypes()

    def generate_dtypes(self):
        dtypes = [
            ("log_likelihood_spectra", float),
            ("Chi square_spectra", float),
            ("log_likelihood_lifetime", float),
            ("log_likelihood_max_abs_pos", float),
            ("max_abs_pos", float),
        ]
        for temp in self.temperature_list:
            dtypes.append((f"Ex_kr_{temp}K", float))
            dtypes.append((f"Ex_knr_{temp}K", float))
        return dtypes

    def update_with_model_config(self, model_config):
        super().update_with_model_config(model_config)
        self.check_available_lifetime()
        self.dtypes = self.generate_dtypes()

    def check_available_lifetime(self):
        temperature_lifetimes_exp = {}
        # change the keys of the dictionary to integers
        self.temperature_lifetimes_exp = {
            float(k): v for k, v in self.temperature_lifetimes_exp.items()
        }
        for temp, exp_lifetime in self.temperature_lifetimes_exp.items():
            index = np.argwhere(self.temperature_list == temp)
            if len(index) == 0:
                logging.debug(f"Temperature {temp} not found in the data")
            else:
                logging.debug(f"index: {index}")
                logging.debug(f"Temperature {temp} found in the data")
                temperature_lifetimes_exp[temp] = exp_lifetime
        self.temperature_lifetimes_exp = temperature_lifetimes_exp.copy()
        logging.debug(
            f"temperature_lifetimes_exp: {temperature_lifetimes_exp}"
        )
        self.error_in_lifetimes = {
            temp: self.relative_error_lifetime * lifetime
            for temp, lifetime in self.temperature_lifetimes_exp.items()
        }

    def log_probability(self, theta, exp_data, inv_co_var_mat_pl):
        """This is an example of a log probability function for the model."""
        lp = self.log_prior(theta)
        return_out = (-np.inf,) + len(self.dtypes) * (None,)
        if lp == -np.inf:
            return return_out
        log_like, chi_squared, data = self.log_likelihood(
            theta,
            exp_data,
            inv_co_var_mat_pl,
        )
        log_prob = lp + log_like[0][0]
        # sum all the errors
        max_abs_pos_error = (
            data.D.hw[data.D.alpha[:, -1].argmax()] - self.max_abs_pos_exp
        ) ** 2 / (2 * self.error_in_max_abs_pos**2)
        log_prob -= max_abs_pos_error
        lifetime_errors = []
        log_likelihood_lifetime = 0
        for temp, exp_lifetime in self.temperature_lifetimes_exp.items():
            index = np.argwhere(self.temperature_list == temp)[0][0]
            calculated_lifetime = 1 / (
                data.EX.knr[0][index] + data.EX.kr[index]
            )
            error_in_lifetime = self.error_in_lifetimes[temp]
            lifetime_error = (calculated_lifetime - exp_lifetime) ** 2 / (
                2 * error_in_lifetime**2
            )
            log_likelihood_lifetime += lifetime_error
            log_prob -= lifetime_error
            lifetime_errors.append(calculated_lifetime)
        if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
            return return_out
        return_out = (
            log_prob,  # everything added<--
            log_like[0],  # PL spectra
            chi_squared[0][0],
            log_likelihood_lifetime,
            max_abs_pos_error,
            data.D.hw[data.D.alpha[:, -1].argmax()],  # max_abs_pos
        )

        for ii in range(len(data.EX.kr)):
            return_out += (
                data.EX.knr[0][ii],
                data.EX.kr[ii],
            )
        return return_out
