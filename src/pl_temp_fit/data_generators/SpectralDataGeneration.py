"""Class containing how to generate data for experimental data fitting."""

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from pl_temp_fit.model_function import LTL


class SpectralDataGeneration:
    """Class containing how to generate data for experimental data fitting.

    Attributes
    ----------
    temperature_list : list
        List of temperature values.
    hws_list : list
        List of photon energies for the spectras.
    error_temperature_sigma : float
        Standard deviation of the temperature error.
    error_intensity_sigma : float
        Standard deviation of the intensity error.
    error_hws_sigma : float
        Standard deviation of the hws error.

    """

    def __init__(self, temperature_list, hws_list):
        self.temperature_list = temperature_list
        self.hws_list = hws_list
        self.error_temperature_sigma = 0.1
        self.error_intensity_sigma = 0.1
        self.error_hws_sigma = 0.1
        self.luminescence_exp = "PL"
        self.lowsignallimit = 0.03
        self.noise_sigma = 0.01
        self.dtypes = [
            ("log_likelihood_spectra", float),
            ("Chi square_spectra", float),
            ("Ex_kr", float),
            ("Ex_knr", float),
        ]

    def update_with_model_config(self, model_config):
        """Update the model with the model configuration."""
        self.error_hws_sigma = model_config["hws_std_err"]
        self.error_intensity_sigma = model_config[
            "relative_intensity_std_error_pl"
        ]
        self.error_temperature_sigma = model_config["Temp_std_err"]
        self.noise_sigma = model_config["noise_sigma"]
        self.min_bounds = model_config["min_bounds"]
        self.max_bounds = model_config["max_bounds"]
        self.fixed_parameters_dict = model_config["fixed_parameters_dict"]
        self.params_to_fit_init = model_config["params_to_fit_init"]

    def add_relative_intensity_error(self, model_data) -> np.array:
        """Add relative intensity error to the model data."""
        relative_intensity_model = np.max(model_data, axis=0) / max(
            model_data.reshape(-1, 1)
        )
        relative_intensity_model_error = (
            relative_intensity_model
            + np.random.normal(
                0, self.error_intensity_sigma, len(relative_intensity_model)
            )
        )
        relative_intensity_model_error = np.abs(
            relative_intensity_model_error
            / np.max(relative_intensity_model_error)
        )
        model_data = (
            model_data
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data

    def generate_data(self, params_to_fit):
        """Run the model to generate the   spectra.

        Args:
        ----
        temperature_list_pl (np.array): The temperature list for the PL spectra
        hws_pl (np.array): The photon energies for the PL spectra
        fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
        params_to_fit (dict): The parameters to fit in the model

        Returns:
        -------
        tuple: The model data for the PL spectra and the radiative and non-radiative recombination rates

        """
        data = LTL.Data()
        data.update(**self.fixed_parameters_dict)
        data.update(**params_to_fit)
        data.D.Luminecence_exp = self.luminescence_exp
        # error in the temperature of the sample
        temperature_list = self.temperature_list + np.random.normal(
            0, self.error_temperature_sigma, len(self.temperature_list)
        )
        # error in the detection wavelength
        hws_list = self.hws_list + np.random.normal(
            0, self.error_hws_sigma, len(self.hws_list)
        )
        data.D.T = temperature_list
        LTL.ltlcalc(data)
        model_data = data.D.kr_hw
        model_data_interp = np.zeros(
            (len(self.hws_list), len(self.temperature_list))
        )
        for i in range(len(self.temperature_list)):
            model_data_interp[:, i] = np.interp(
                hws_list, data.D.hw, model_data[:, i]
            )
        model_data_interp = self.add_relative_intensity_error(
            model_data_interp
        )
        return model_data_interp, data

    def log_likelihood(
        self,
        theta,
        exp_data,
        inv_co_var_mat_pl,
    ):
        """Calculate the log likelihood and the chi squared of the model data and the experimental data.

        Args:
        ----
        theta (np.array): The parameters for the model
        data (np.array): The experimental data for the  spectra
        inv_co_var_mat_pl (np.array): The inverse of the covariance matrix for the  spectra
        temperature_list_pl (np.array): The temperature list for the  spectra
        hws_pl (np.array): The photon energies for the  spectra
        fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
        params_to_fit (dict): The parameters to fit in the model

        Returns:
        -------
        float: The log likelihood of the model data and the experimental data

        """
        params_to_fit_updated = self.update_params_to_fit(theta)
        model_data, data_model = self.generate_data(params_to_fit_updated)
        model_data = model_data / np.max(model_data.reshape(-1, 1))
        exp_data = exp_data / np.max(exp_data.reshape(-1, 1))
        diff = exp_data.reshape(-1, 1) - model_data.reshape(-1, 1)
        diff[np.abs(exp_data.reshape(-1, 1)) < self.lowsignallimit] = 0
        log_likelihood = -0.5 * np.dot(diff.T, np.dot(inv_co_var_mat_pl, diff))
        chi_squared = (
            -2 * log_likelihood / (len(exp_data.reshape(-1, 1)) - len(theta))
        )
        return log_likelihood, chi_squared, data_model

    def log_probability(
        self,
        theta,
        data_pl,
        inv_co_var_mat_pl,
    ):
        """This is an example of a log probability function for the model."""
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return -np.inf, None, None, None, None
        log_like, chi_squared, data = self.log_likelihood(
            theta,
            data_pl,
            inv_co_var_mat_pl,
        )
        log_prob = lp + log_like[0][0]
        if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
            return -np.inf, None, None, None, None
        return (
            log_prob,
            log_like[0],
            chi_squared[0][0],
            data.EX.knr[0][-1],
            data.EX.kr[-1],
        )

    def update_params_to_fit(self, theta):
        """Update the parameters to fit with the new values of theta."""
        params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}, "I0": {}}
        counter = 0
        try:
            for key in ["EX", "CT", "D", "I0"]:
                if self.params_to_fit_init[key] == {}:
                    continue
                for id, key2 in enumerate(self.params_to_fit_init[key].keys()):
                    params_to_fit_updated[key][key2] = theta[counter]
                    counter += 1
        except ValueError:
            print("The parameters to fit are not in the correct format")
        return params_to_fit_updated

    def get_covariance_matrix(
        self,
        numnber_of_samples=20,
    ):
        """Get the covariance matrix for the model data.

        Args:
        ----
            params_to_fit (dict): The parameters to fit in the model.
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes.
            numnber_of_samples (int): The number of samples to plot.

        Returns:
        -------
            np.array: The covariance matrix for the model data.
            np.array: The variance of the model data.


        """
        true_model_list = []

        for x in range(numnber_of_samples):
            model_data_pl, _ = self.generate_data(self.params_to_fit_init)
            true_model_list.append(
                model_data_pl / np.max(model_data_pl.reshape(-1, 1))
            )

        co_var_mat_test_pl, variance_pl = self._get_covariance_matrix(
            true_model_list, numnber_of_samples
        )
        return co_var_mat_test_pl, variance_pl

    def _get_covariance_matrix(
        self, true_model_signal_list, numnber_of_samples
    ):
        """Get the covariance matrix for the model data.

        Args:
        ----
            true_model_signal_list: {list} --
                list of the model data.
            numnber_of_samples:
                 {int} -- number of samples.

        Returns:
        -------
            tuple: The covariance matrix and the variance of the model data.

        """
        variance = (
            np.var(np.array(true_model_signal_list), axis=0) + self.noise_sigma
        )
        co_var_mat_test = np.cov(
            np.array(true_model_signal_list).reshape(numnber_of_samples, -1),
            rowvar=False,
        )
        np.fill_diagonal(
            co_var_mat_test, co_var_mat_test.diagonal() + self.noise_sigma
        )
        return co_var_mat_test, variance

    def log_prior(self, theta):
        """Calculate the log prior for the parameters.

        Args:
        ----
        theta (np.array): The parameters to fit in the model

        Returns:
        -------
        float: The log prior for the parameters

        """
        counter = 0
        for param_key in ["EX", "CT", "D", "I0"]:
            if self.min_bounds[param_key] == {}:
                continue

            for id, key in enumerate(self.min_bounds[param_key].keys()):
                if (
                    self.min_bounds[param_key][key] > theta[counter]
                    or self.max_bounds[param_key][key] < theta[counter]
                ):
                    return -np.inf
                counter += 1
        return 0.0

    def get_maximum_likelihood_estimate(
        self,
        exp_data_pl,
        co_var_mat_pl,
        save_folder,
        coeff_spread=0.1,
        num_coords=5,
    ):
        def nll(*args):
            nll = -self.log_probability(*args)[0]
            # check type of nll
            if isinstance(nll, np.ndarray):
                return nll[0][0]
            return nll

        coords, max_bound_list, min_bound_list, num_parameters = (
            self.get_init_coords(
                coeff_spread,
                num_coords,
            )
        )
        min_fun = np.inf
        print("running the minimisation")
        inv_covar_mat = np.linalg.inv(co_var_mat_pl)
        for i, coord in enumerate(coords):
            print(f"step {i}")
            soln = minimize(
                nll,
                coord,
                args=(
                    exp_data_pl,
                    inv_covar_mat,
                ),
                bounds=[
                    (min_bound_list[i], max_bound_list[i])
                    for i in range(num_parameters)
                ],
                tol=1e-2,
            )
            if "NORM_OF_PROJECTED_GRADIENT_" in soln.message:
                print("NORM_OF_PROJECTED_GRADIENT_")
                print(soln.x)
                print(soln.fun)
                continue
            if soln.fun < min_fun:
                min_fun = soln.fun
                soln_min = soln
        print(soln_min.x)
        print("Maximum likelihood estimates:")
        counter = 0
        for key in ["EX", "CT", "D"]:
            if self.params_to_fit_init[key] == {}:
                continue
            for key2 in self.params_to_fit_init[key].keys():
                print(f"  {key}_{key2} = {soln_min.x[counter]:.3f}")
                counter += 1
        print("Maximum log likelihood:", soln.fun)
        # Calculate the Fisher Information Matrix
        # Calculate the confidence intervals
        confidence_level = 0.99
        confidence_intervals = self.get_confidence_intervals(
            soln_min, confidence_level=confidence_level
        )
        # print those into a file
        with open(save_folder + "/maximum_likelihood_estimate.txt", "w") as f:
            f.write("Maximum likelihood estimates:\n")
            counter = 0
            for key in ["EX", "CT", "D", "I0"]:
                if self.params_to_fit_init[key] == {}:
                    continue
                for key2 in self.params_to_fit_init[key].keys():
                    f.write(f"  {key}_{key2} = {soln_min.x[counter]:.3f}\n")
                    f.write(
                        f"  Confidence interval confidence_level {confidence_level}: {confidence_intervals[counter]} \n"
                    )
                    counter += 1

            f.write(f"Maximum log likelihood: {soln_min.fun}\n")
        return soln_min

    def get_confidence_intervals(self, soln_min, confidence_level=0.99):
        # Calculate the inverse of the Hessian matrix
        inv_hessian = soln_min.hess_inv.todense()
        logging.debug(f"inv_hessian: {inv_hessian}")
        # Calculate the standard errors (square roots of the diagonal elements)
        standard_errors = np.sqrt(np.diag(inv_hessian))

        # Calculate the z-score for the given confidence level
        z_score = norm.ppf(1 - (confidence_level) / 2)

        # Calculate the confidence intervals
        confidence_intervals = []
        for i in range(len(standard_errors)):
            confidence_intervals.append(
                (
                    soln_min.x[i] - z_score * standard_errors[i],
                    soln_min.x[i] + z_score * standard_errors[i],
                )
            )

        return confidence_intervals

    def get_init_coords(self, coeff_spread, num_coords):
        """Get the initial coordinates for the simple fit."""
        init_params, min_bound_list, max_bound_list = [], [], []
        counter = 0
        for key in ["EX", "CT", "D", " I0"]:
            if self.params_to_fit_init[key] == {}:
                continue
            for key2 in self.params_to_fit_init[key].keys():
                init_params.append(self.params_to_fit_init[key][key2])
                min_bound_list.append(self.min_bounds[key][key2])
                max_bound_list.append(self.max_bounds[key][key2])
                counter += 1
        min_bound_list = np.array(min_bound_list)
        max_bound_list = np.array(max_bound_list)
        min_bound_list = np.array(min_bound_list)
        max_bound_list = np.array(max_bound_list)
        num_parameters = counter
        coords = init_params + 0.1 * coeff_spread * (
            max_bound_list - min_bound_list
        ) * np.random.randn(num_coords, num_parameters)
        return coords, max_bound_list, min_bound_list, num_parameters
