"""Class containing how to generate data for experimental data fitting."""

import numpy as np

from pl_temp_fit.model_function import LTL


def log_prior(theta, min_bounds, max_bounds):
    """Calculate the log prior for the parameters.

    Args:
    ----
    theta (np.array): The parameters to fit
    min_bounds (dict): The minimum bounds for the parameters
    max_bounds (dict): The maximum bounds for the parameters

    Returns:
    -------
    float: The log prior for the parameters

    """
    counter = 0
    for param_key in ["EX", "CT", "D"]:
        if min_bounds[param_key] == {}:
            continue

        for id, key in enumerate(min_bounds[param_key].keys()):
            if (
                min_bounds[param_key][key] > theta[counter]
                or max_bounds[param_key][key] < theta[counter]
            ):
                return -np.inf
            counter += 1
    return 0.0


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

    def generate_data(self, fixed_parameters_dict, params_to_fit):
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
        data.update(**fixed_parameters_dict)
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
            (len(self.hws_pl), len(self.temperature_list_pl))
        )
        for i in range(len(self.temperature_list_pl)):
            model_data_interp[:, i] = np.interp(
                hws_list, data.D.hw, model_data[:, i]
            )
        EX_kr = data.EX.kr
        Ex_knr = data.EX.knr
        model_data_interp = self.add_relative_intensity_error(
            model_data_interp
        )
        return model_data_interp, EX_kr, Ex_knr

    def log_likelihood(
        self,
        theta,
        data,
        inv_co_var_mat_pl,
        fixed_parameters_dict={},
        params_to_fit={},
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
        params_to_fit_updated = self.update_params_to_fit(theta, params_to_fit)
        model_data, _, _ = self.generate_data(
            fixed_parameters_dict, params_to_fit_updated
        )
        model_data = model_data / np.max(model_data).reshape(-1, 1)
        data = data / np.max(data).reshape(-1, 1)
        diff = data.reshape(-1, 1) - model_data.reshape(-1, 1)
        diff[np.abs(data) < self.lowsignallimit] = 0
        log_likelihood = -0.5 * np.dot(diff.T, np.dot(inv_co_var_mat_pl, diff))
        chi_squared = -2 * log_likelihood / (len(data) - len(theta))
        return log_likelihood, chi_squared

    def log_probability_pl(
        self,
        theta,
        data_pl,
        inv_co_var_mat_pl,
        fixed_parameters_dict,
        params_to_fit,
        min_bounds,
        max_bounds,
    ):
        lp = log_prior(theta, min_bounds, max_bounds)
        if lp == -np.inf:
            return -np.inf, None, None, None, None
        log_like, Chi_squared, EX_kr, Ex_knr = self.log_likelihood(
            theta,
            data_pl,
            inv_co_var_mat_pl,
            fixed_parameters_dict,
            params_to_fit,
        )
        log_prob = lp + log_like[0][0]
        if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
            return -np.inf, None, None, None, None
        return (
            log_prob,
            log_like[0],
            Chi_squared[0][0],
            EX_kr[-1],
            Ex_knr[0][-1],
        )

    def update_params_to_fit(self, theta, params_to_fit):
        """Update the parameters to fit with the new values of theta."""
        params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}}
        counter = 0
        try:
            for key in ["EX", "CT", "D"]:
                if params_to_fit[key] == {}:
                    continue
                for id, key2 in enumerate(params_to_fit[key].keys()):
                    params_to_fit_updated[key][key2] = theta[counter]
                    counter += 1
        except ValueError:
            print("The parameters to fit are not in the correct format")
        return params_to_fit_updated
