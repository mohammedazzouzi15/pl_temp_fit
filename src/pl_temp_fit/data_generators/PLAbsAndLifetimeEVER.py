"""Class to fit the PL data with a model that includes the absorption peak and lifetime parameters for all temps HY 08/10/2024."""

import logging

import numpy as np

from pl_temp_fit.data_generators.SpectralDataGeneration import (
    SpectralDataGeneration,
)


class PLAbsAndLifetimeEVER(SpectralDataGeneration):
    def __init__(self, temperature_list, hws_list):
        super().__init__(temperature_list, hws_list)
        self.dtypes = [
            ("log_likelihood_spectra", float),
            ("Chi square_spectra", float),
            ("Ex_kr_300K", float),
            ("Ex_knr_300K", float),
            ("Ex_kr_260K", float),
            ("Ex_knr_260K", float),
            ("Ex_kr_220K", float),
            ("Ex_knr_220K", float),
            ("Ex_kr_180K", float),
            ("Ex_knr_180K", float),
            ("Ex_kr_140K", float),
            ("Ex_knr_140K", float),
            ("Ex_kr_100K", float),
            ("Ex_knr_100K", float),
            ("Ex_kr_60K", float),
            ("Ex_knr_60K", float),
            ("max_abs_pos", float),
        ]
        self.max_abs_pos_exp = 1.5  # in eV
        self.error_in_max_abs_pos = 0.01  # in eV
        self.relative_error_lifetime = 0.05  # 5% relative error
        self.temperature_lifetimes_exp = {
            60: 1e-9,
            100: 1e-9,
            140: 1e-9,
            180: 1e-9,
            220: 1e-9,
            260: 1e-9,
            300: 1e-9,
        }  # in seconds
        self.error_in_lifetimes = {
            temp: self.relative_error_lifetime * lifetime
            for temp, lifetime in self.temperature_lifetimes_exp.items()
        }
        self.check_available_lifetime()

    def update_with_model_config(self, model_config):
        super().update_with_model_config(model_config)
        self.check_available_lifetime()

    def check_available_lifetime(self):
        temperature_lifetimes_exp = {}
        # change the keys of the dictionary to integers
        self.temperature_lifetimes_exp = {
            int(k): v for k, v in self.temperature_lifetimes_exp.items()
        }
        for temp, exp_lifetime in self.temperature_lifetimes_exp.items():
            index = np.argwhere(self.temperature_list == temp)
            if index is None:
                logging.debug(f"Temperature {temp} not found in the data")
            else:
                logging.debug(f"index: {index}")
                logging.debug(f"Temperature {temp} found in the data")
                temperature_lifetimes_exp[temp] = exp_lifetime
        self.temperature_lifetimes_exp = temperature_lifetimes_exp

    def log_probability(self, theta, exp_data, inv_co_var_mat_pl):
        """This is an example of a log probability function for the model."""
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return (
                -np.inf,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )  # do not forget to return the correct number of outputs

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

        for temp, exp_lifetime in self.temperature_lifetimes_exp.items():
            index = np.argwhere(self.temperature_list == temp)[0][0]
            calculated_lifetime = 1 / (
                data.EX.knr[0][index] + data.EX.kr[index]
            )
            error_in_lifetime = self.error_in_lifetimes[temp]
            lifetime_error = (calculated_lifetime - exp_lifetime) ** 2 / (
                2 * error_in_lifetime**2
            )
            log_prob -= lifetime_error
            lifetime_errors.append(calculated_lifetime)

        if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
            return (
                -np.inf,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        return (
            log_prob,  # everything added<--
            log_like[0],  # PL spectra
            chi_squared[0][0],
            data.EX.knr[0][-1],  # 300K
            data.EX.kr[-1],
            data.EX.knr[0][-2],  # 260K
            data.EX.kr[-2],
            data.EX.knr[0][-3],  # 220K
            data.EX.kr[-3],
            data.EX.knr[0][-4],  # 180K
            data.EX.kr[-4],
            data.EX.knr[0][-5],  # 140K
            data.EX.kr[-5],
            data.EX.knr[0][-6],  # 100K
            data.EX.kr[-6],
            data.EX.knr[0][-7],  # 60K
            data.EX.kr[-7],  # apparently dtypes don't like lists..
            data.D.hw[data.D.alpha[:, -1].argmax()],
        )  # do not forget to return the correct number of outputs
