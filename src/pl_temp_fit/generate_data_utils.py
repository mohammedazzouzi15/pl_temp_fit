from scipy.optimize import minimize
import numpy as np
from pl_temp_fit import LTL
import os


def generate_data(
    temperature_list_EL,
    hws_EL,
    temperature_list_PL,
    hws_PL,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_PL,
    relative_intensity_std_error_EL,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
):
    """Generate the data for the EL and PL spectra with added noise

    Args:

            temperature_list_EL (np.array): The temperature list for the EL spectra
            hws_EL (np.array): The photon energies for the EL spectra
            temperature_list_PL (np.array): The temperature list for the PL spectra
            hws_PL (np.array): The photon energies for the PL spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_PL (float): The standard deviation of the relative intensity error for the PL spectra
            relative_intensity_std_error_EL (float): The standard deviation of the relative intensity error for the EL spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the PL and EL spectra and the true parameters
    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_PL, relative_intensity_std_error
    ):
        relative_intensity_model = np.max(model_data_PL, axis=0) / max(
            model_data_PL.reshape(-1, 1)
        )
        relative_intensity_model_error = (
            relative_intensity_model
            + np.random.normal(
                0, relative_intensity_std_error, len(relative_intensity_model)
            )
        )
        relative_intensity_model_error = np.abs(
            relative_intensity_model_error
            / np.max(relative_intensity_model_error)
        )
        model_data_PL = (
            model_data_PL
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_PL

    # error in the temperature of the sample

    temperature_list_PL = temperature_list_PL + np.random.normal(
        0, Temp_std_err, len(temperature_list_PL)
    )
    temperature_list_EL = temperature_list_EL + np.random.normal(
        0, Temp_std_err, len(temperature_list_EL)
    )
    # error in the detection wavelength
    hws_PL = hws_PL + np.random.normal(0, hws_std_err, len(hws_PL))
    hws_EL = hws_EL + np.random.normal(0, hws_std_err, len(hws_EL))
    model_data_EL, model_data_PL = el_trial(
        temperature_list_EL,
        hws_EL,
        temperature_list_PL,
        hws_PL,
        fixed_parameters_dict,
        params_to_fit,
    )
    model_data_PL = add_relative_intensity_error(
        model_data_PL, relative_intensity_std_error_PL
    )
    model_data_EL = add_relative_intensity_error(
        model_data_EL, relative_intensity_std_error_EL
    )

    return model_data_PL, model_data_EL


def generate_data_PL(
    temperature_list_PL,
    hws_PL,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_PL,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
    **kwargs,
):
    """Generate the data for the PL spectra with added noise

    Args:


            temperature_list_PL (np.array): The temperature list for the PL spectra
            hws_PL (np.array): The photon energies for the PL spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_PL (float): The standard deviation of the relative intensity error for the PL spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the PL and EL spectra and the true parameters
    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_PL, relative_intensity_std_error
    ):
        relative_intensity_model = np.max(model_data_PL, axis=0) / max(
            model_data_PL.reshape(-1, 1)
        )
        relative_intensity_model_error = (
            relative_intensity_model
            + np.random.normal(
                0, relative_intensity_std_error, len(relative_intensity_model)
            )
        )
        relative_intensity_model_error = np.abs(
            relative_intensity_model_error
            / np.max(relative_intensity_model_error)
        )
        model_data_PL = (
            model_data_PL
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_PL

    # error in the temperature of the sample

    temperature_list_PL = temperature_list_PL + np.random.normal(
        0, Temp_std_err, len(temperature_list_PL)
    )
    # error in the detection wavelength
    hws_PL = hws_PL + np.random.normal(0, hws_std_err, len(hws_PL))
    model_data_PL = pl_trial(
        temperature_list_PL,
        hws_PL,
        fixed_parameters_dict,
        params_to_fit,
    )
    model_data_PL = add_relative_intensity_error(
        model_data_PL, relative_intensity_std_error_PL
    )
    return model_data_PL


def set_parameters(data, fixed_parameters_dict):
    """Set the fixed parameters for the model that are not the same as the default

    Args:
            data (LTL.Data): The data object for the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            LTL.Data: The data object for the model with the fixed parameters set
    """
    for key, value in fixed_parameters_dict.items():
        data[key].update(value)
    return data


def pl_trial(
    temperature_list_PL,
    hws_PL,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model for the  PL spectra"""
    data = LTL.Data()
    data.update(**fixed_parameters_dict)
    data.update(**params_to_fit)
    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_PL  # np.array([300.0, 150.0, 80.0])
    LTL.LTLCalc(data)
    PL_results = data.D.kr_hw  # .reshape(-1, 1)
    PL_results_interp = np.zeros((len(hws_PL), len(temperature_list_PL)))
    for i in range(len(temperature_list_PL)):
        PL_results_interp[:, i] = np.interp(
            hws_PL, data.D.hw, PL_results[:, i]
        )
    return PL_results_interp  # / max(PL_results)


def el_trial(
    temperature_list_EL,
    hws_EL,
    temperature_list_PL,
    hws_PL,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model for the EL and PL spectra"""
    data = LTL.Data()
    data.update(**fixed_parameters_dict)
    data.update(**params_to_fit)
    data.D.Luminecence_exp = "EL"
    LTL.LTLCalc(data)
    EL_results = data.D.kr_hw  # .reshape(-1, 1)
    EL_results_interp = np.zeros((len(hws_EL), len(temperature_list_EL)))
    for i in range(len(temperature_list_EL)):
        EL_results_interp[:, i] = np.interp(
            hws_EL, data.D.hw, EL_results[:, i]
        )

    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_PL  # np.array([300.0, 150.0, 80.0])
    LTL.LTLCalc(data)
    PL_results = data.D.kr_hw  # .reshape(-1, 1)
    PL_results_interp = np.zeros((len(hws_PL), len(temperature_list_PL)))
    for i in range(len(temperature_list_PL)):
        PL_results_interp[:, i] = np.interp(
            hws_PL, data.D.hw, PL_results[:, i]
        )
    return EL_results_interp, PL_results_interp  # / max(PL_results)


def log_prior(theta, min_bounds, max_bounds):
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


def el_loglike(
    theta,
    data_EL,
    data_PL,
    co_var_mat_EL,
    co_var_mat_PL,
    temperature_list_EL,
    hws_EL,
    temperature_list_PL,
    hws_PL,
    fixed_parameters_dict={},
    params_to_fit={},
):
    params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}}
    counter = 0
    try:
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue

            for id, key2 in enumerate(params_to_fit[key].keys()):
                params_to_fit_updated[key][key2] = theta[counter]
                counter += 1

    except Exception as e:
        print(e)
        raise ValueError("The parameters to fit are not in the correct format")
    model_data_EL, model_data_PL = el_trial(
        temperature_list_EL,
        hws_EL,
        temperature_list_PL,
        hws_PL,
        fixed_parameters_dict,
        params_to_fit_updated,
    )
    model_data_EL = model_data_EL / np.max(model_data_EL.reshape(-1, 1))
    model_data_EL = model_data_EL.reshape(-1, 1)
    data_EL = data_EL / np.max(data_EL.reshape(-1, 1))
    data_EL = data_EL.reshape(-1, 1)
    model_data_PL = model_data_PL / np.max(model_data_PL.reshape(-1, 1))
    model_data_PL = model_data_PL.reshape(-1, 1)
    data_PL = data_PL / np.max(data_PL.reshape(-1, 1))
    data_PL = data_PL.reshape(-1, 1)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data_EL).any() or np.isinf(model_data_EL).any():
        # print("NaN in model_data")
        return -np.inf
    diff_EL = data_EL - model_data_EL
    diff_EL[np.abs(diff_EL) < 1e-3] = 0
    diff_EL[np.abs(data_EL) < 3e-2] = 0
    loglike = -0.5 * np.dot(
        diff_EL.T, np.dot(np.linalg.inv(co_var_mat_EL), diff_EL)
    )
    diff_PL = data_PL - model_data_PL
    diff_PL[np.abs(diff_PL) < 1e-3] = 0
    diff_PL[np.abs(data_PL) < 3e-2] = 0
    loglike = loglike - 0.5 * np.dot(
        diff_PL.T, np.dot(np.linalg.inv(co_var_mat_PL), diff_PL)
    )
    return loglike


def log_probability(
    theta,
    data_EL,
    data_PL,
    co_var_mat_EL,
    co_var_mat_PL,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf
    log_like = el_loglike(
        theta,
        data_EL,
        data_PL,
        co_var_mat_EL,
        co_var_mat_PL,
        X["temperature_list_EL"],
        X["hws_EL"],
        X["temperature_list_PL"],
        X["hws_PL"],
        fixed_parameters_dict,
        params_to_fit,
    )
    log_prob = lp + log_like[0]

    # print(f"log_prob is {log_prob}")
    # assert log_prob is a float
    if np.isnan(log_like):
        return -np.inf
    if np.isinf(log_like):
        return -np.inf
    if log_prob is None:

        return -np.inf
    assert (
        log_prob.dtype.kind == "f"
    ), f"the log_prob is not a float but a {type(log_prob)}"

    return log_prob


def pl_loglike(
    theta,
    data_PL,
    co_var_mat_PL,
    temperature_list_PL,
    hws_PL,
    fixed_parameters_dict={},
    params_to_fit={},
):
    params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}}
    counter = 0
    try:
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue

            for id, key2 in enumerate(params_to_fit[key].keys()):
                params_to_fit_updated[key][key2] = theta[counter]
                counter += 1

    except Exception as e:
        print(e)
        raise ValueError("The parameters to fit are not in the correct format")
    model_data_PL = pl_trial(
        temperature_list_PL,
        hws_PL,
        fixed_parameters_dict,
        params_to_fit_updated,
    )
    model_data_PL = model_data_PL / np.max(model_data_PL.reshape(-1, 1))
    model_data_PL = model_data_PL.reshape(-1, 1)
    data_PL = data_PL / np.max(data_PL.reshape(-1, 1))
    data_PL = data_PL.reshape(-1, 1)
    diff_PL = data_PL - model_data_PL
    diff_PL[np.abs(diff_PL) < 1e-3] = 0
    diff_PL[np.abs(data_PL) < 3e-2] = 0
    loglike = -0.5 * np.dot(
        diff_PL.T, np.dot(np.linalg.inv(co_var_mat_PL), diff_PL)
    )
    return loglike


def log_probability_PL(
    theta,
    data_PL,
    co_var_mat_PL,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf
    log_like = pl_loglike(
        theta,
        data_PL,
        co_var_mat_PL,
        X["temperature_list_PL"],
        X["hws_PL"],
        fixed_parameters_dict,
        params_to_fit,
    )
    log_prob = lp + log_like[0]

    # print(f"log_prob is {log_prob}")
    # assert log_prob is a float
    if np.isnan(log_like):
        return -np.inf
    if np.isinf(log_like):
        return -np.inf
    if log_prob is None:

        return -np.inf
    assert (
        log_prob.dtype.kind == "f"
    ), f"the log_prob is not a float but a {type(log_prob)}"

    return log_prob
