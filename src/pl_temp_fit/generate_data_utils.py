"""Functions to generate the data for the EL and PL spectra with added noise and the functions to fit the data using the model.

The data is generated using the model and the parameters are fitted using the data.
The functions in this module are used to generate the data for the EL and PL spectra with added noise.
the function in this module are:
    - generate_data
    - generate_data_pl
    - set_parameters
    - pl_trial
    - el_trial
    - log_prior
    - el_loglike
    - log_probability
    - pl_loglike
    - log_probability_pl
"""

import numpy as np

from pl_temp_fit.model_function import LTL


def generate_data(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_pl,
    relative_intensity_std_error_el,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
):
    """Generate the data for the EL and PL spectra with added noise.

    Args:
    ----
            temperature_list_el (np.array): The temperature list for the EL spectra
            hws_el (np.array): The photon energies for the EL spectra
            temperature_list_pl (np.array): The temperature list for the PL spectra
            hws_pl (np.array): The photon energies for the PL spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_pl (float): The standard deviation of the relative intensity error for the PL spectra
            relative_intensity_std_error_el (float): The standard deviation of the relative intensity error for the EL spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the PL and EL spectra and the true parameters

    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error
    ) -> np.array:
        relative_intensity_model = np.max(model_data_pl, axis=0) / max(
            model_data_pl.reshape(-1, 1)
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
        model_data_pl = (
            model_data_pl
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_pl

    # error in the temperature of the sample

    temperature_list_pl = temperature_list_pl + np.random.normal(
        0, Temp_std_err, len(temperature_list_pl)
    )
    temperature_list_el = temperature_list_el + np.random.normal(
        0, Temp_std_err, len(temperature_list_el)
    )
    # error in the detection wavelength
    hws_pl = hws_pl + np.random.normal(0, hws_std_err, len(hws_pl))
    hws_el = hws_el + np.random.normal(0, hws_std_err, len(hws_el))
    model_data_el, model_data_pl, dv_nr, CT_knr, CT_kr, EX_knr, EX_kr = (
        el_trial(
            temperature_list_el,
            hws_el,
            temperature_list_pl,
            hws_pl,
            fixed_parameters_dict,
            params_to_fit,
        )
    )
    model_data_pl = add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error_pl
    )
    model_data_el = add_relative_intensity_error(
        model_data_el, relative_intensity_std_error_el
    )

    return model_data_pl, model_data_el


def generate_data_pl(
    temperature_list_pl,
    hws_pl,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_pl,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
    **kwargs,
):
    """Generate the data for the PL spectra with added noise.

    Args:
    ----
            temperature_list_pl (np.array): The temperature list for the PL spectra
            hws_pl (np.array): The photon energies for the PL spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_pl (float): The standard deviation of the relative intensity error for the PL spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the PL and EL spectra and the true parameters

    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error
    ) -> np.array:
        relative_intensity_model = np.max(model_data_pl, axis=0) / max(
            model_data_pl.reshape(-1, 1)
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
        model_data_pl = (
            model_data_pl
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_pl

    # error in the temperature of the sample

    temperature_list_pl = temperature_list_pl + np.random.normal(
        0, Temp_std_err, len(temperature_list_pl)
    )
    # error in the detection wavelength
    hws_pl = hws_pl + np.random.normal(0, hws_std_err, len(hws_pl))
    model_data_pl, EX_kr, Ex_knr = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit,
    )
    model_data_pl = add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error_pl
    )
    return model_data_pl, EX_kr, Ex_knr


def set_parameters(data, fixed_parameters_dict):
    """Set the fixed parameters for the model that are not the same as the default.

    Args:
    ----
            data (LTL.Data): The data object for the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            LTL.Data: The data object for the model with the fixed parameters set

    """
    for key, value in fixed_parameters_dict.items():
        data[key].update(value)
    return data


def pl_trial(
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model to generate the  PL spectra.

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
    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_pl  # np.array([300.0, 150.0, 80.0])
    LTL.ltlcalc(data)
    pl_results = data.D.kr_hw  # .reshape(-1, 1)
    pl_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
    for i in range(len(temperature_list_pl)):
        pl_results_interp[:, i] = np.interp(
            hws_pl, data.D.hw, pl_results[:, i]
        )
    EX_kr = data.EX.kr
    Ex_knr = data.EX.knr
    return pl_results_interp, EX_kr, Ex_knr


def el_trial(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model to generate the EL and PL spectra.

    Args:
    ----
    temperature_list_el (np.array): The temperature list for the EL spectra
    hws_el (np.array): The photon energies for the EL spectra
    temperature_list_pl (np.array): The temperature list for the PL spectra
    hws_pl (np.array): The photon energies for the PL spectra
    fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    params_to_fit (dict): The parameters to fit in the model

    Returns:
    -------
    tuple: The model data for the EL and PL spectra

    """
    data = LTL.Data()
    data.update(**fixed_parameters_dict)
    data.update(**params_to_fit)
    data.D.Luminecence_exp = "EL"
    LTL.ltlcalc(data)
    el_results = data.D.kr_hw  # .reshape(-1, 1)
    el_results_interp = np.zeros((len(hws_el), len(temperature_list_el)))
    for i in range(len(temperature_list_el)):
        el_results_interp[:, i] = np.interp(
            hws_el, data.D.hw, el_results[:, i]
        )
    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_pl  # np.array([300.0, 150.0, 80.0])
    LTL.ltlcalc(data)
    pl_results = data.D.kr_hw  # .reshape(-1, 1)
    pl_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
    for i in range(len(temperature_list_pl)):
        pl_results_interp[:, i] = np.interp(
            hws_pl, data.D.hw, pl_results[:, i]
        )
    data.get_delta_voc_nr()
    CT_kr = data.CT.kr
    CT_knr = data.CT.knr
    EX_kr = data.EX.kr
    EX_knr = data.EX.knr
    return (
        el_results_interp,
        pl_results_interp,
        data.voltage_results["delta_voc_nr"],
        CT_knr,
        CT_kr,
        EX_knr,
        EX_kr,
    )  # / max(pl_results)


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


def el_loglike(
    theta,
    data_el,
    data_pl,
    inv_co_var_mat_el,
    inv_co_var_mat_pl,
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Calculate the log likelihood for the EL and PL spectra.

    Args:
    ----
    theta (np.array): The parameters to fit
    data_el (np.array): The data for the EL spectra
    data_pl (np.array): The data for the PL spectra
    inv_co_var_mat_el (np.array): The inverse covariance matrix for the EL spectra
    inv_co_var_mat_pl (np.array): The inverse covariance matrix for the PL spectra
    temperature_list_el (np.array): The temperature list for the EL spectra
    hws_el (np.array): The photon energies for the EL spectra
    temperature_list_pl (np.array): The temperature list for the PL spectra
    hws_pl (np.array): The photon energies for the PL spectra
    fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    params_to_fit (dict): The parameters to fit in the model

    Returns:
    -------
    float: The log likelihood for the EL and PL spectra

    """
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

    model_data_el, model_data_pl, dv_nr, CT_knr, CT_kr, EX_knr, EX_kr = (
        el_trial(
            temperature_list_el,
            hws_el,
            temperature_list_pl,
            hws_pl,
            fixed_parameters_dict,
            params_to_fit_updated,
        )
    )
    model_data_el = model_data_el / np.max(model_data_el.reshape(-1, 1))
    model_data_el = model_data_el.reshape(-1, 1)
    data_el = data_el / np.max(data_el.reshape(-1, 1))
    data_el = data_el.reshape(-1, 1)
    model_data_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    model_data_pl = model_data_pl.reshape(-1, 1)
    data_pl = data_pl / np.max(data_pl.reshape(-1, 1))
    data_pl = data_pl.reshape(-1, 1)
    # check that the data in model_data does not contain NaNs or infs
    #if np.isnan(model_data_el).any() or np.isinf(model_data_el).any():
        #return [[-np.inf]], [[10]], [[10]], None, None, None, None
    diff_el = data_el - model_data_el
    diff_el[np.abs(diff_el) < 1e-3] = 0
    diff_el[np.abs(data_el) < 3e-2] = 0
    loglike = -0.5 * np.dot(diff_el.T, np.dot(inv_co_var_mat_el, diff_el))
    diff_pl = data_pl - model_data_pl
    diff_pl[np.abs(diff_pl) < 1e-3] = 0
    diff_pl[np.abs(data_pl) < 3e-2] = 0
    log_like = loglike - 0.5 * np.dot(
        diff_pl.T, np.dot(inv_co_var_mat_pl, diff_pl)
    )
    Chi_squared = -2 * log_like / (len(data_pl) + len(data_el) - len(theta))
    return log_like, Chi_squared, dv_nr, CT_knr, CT_kr, EX_knr, EX_kr


def log_probability(
    theta,
    data_el,
    data_pl,
    inv_co_var_mat_el,
    inv_co_var_mat_pl,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    """Calculate the log probability for the EL and PL spectra.

    Args:
    ----
    theta (np.array): The parameters to fit
    data_el (np.array): The data for the EL spectra
    data_pl (np.array): The data for the PL spectra
    inv_co_var_mat_el (np.array): The inverse covariance matrix for the EL spectra
    inv_co_var_mat_pl (np.array): The inverse covariance matrix for the PL spectra
    X (dict): The data for the EL and PL spectra
    fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    params_to_fit (dict): The parameters to fit in the model
    min_bounds (dict): The minimum bounds for the parameters
    max_bounds (dict): The maximum bounds for the parameters

    Returns:
    -------
    float: The log probability for the EL and PL spectra
    """
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf, -np.inf, 10, None, None, None, None, None
    log_like, chi_squared, dv_nr, CT_knr, CT_kr, EX_knr, EX_kr = el_loglike(
        theta,
        data_el,
        data_pl,
        inv_co_var_mat_el,
        inv_co_var_mat_pl,
        X["temperature_list_el"],
        X["hws_el"],
        X["temperature_list_pl"],
        X["hws_pl"],
        fixed_parameters_dict,
        params_to_fit,
    )
    log_prob = lp + log_like[0][0]
    # Check for invalid log likelihood values
    if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
        return -np.inf, -np.inf, 10, None, None, None, None, None
    return (
        log_prob,
        log_like[0][0],
        chi_squared[0][0],
        dv_nr[0][-1],
        CT_knr[0][-1],
        CT_kr[-1],
        EX_knr[0][-1],
        EX_kr[-1],
    )


def pl_loglike(
    theta,
    data_pl,
    inv_co_var_mat_pl,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Calculate the log likelihood for the PL spectra.

    Args:
    ----
    theta (np.array): The parameters to fit
    data_pl (np.array): The data for the PL spectra
    inv_co_var_mat_pl (np.array): The inverse covariance matrix for the PL spectra
    temperature_list_pl (np.array): The temperature list for the PL spectra
    hws_pl (np.array): The photon energies for the PL spectra
    fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    params_to_fit (dict): The parameters to fit in the model

    Returns:
    -------
    float: The log likelihood for the PL spectra

    """
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
    model_data_pl, EX_kr, Ex_knr = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit_updated,
    )
    model_data_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    model_data_pl = model_data_pl.reshape(-1, 1)
    data_pl_copy = data_pl.copy()
    data_pl_copy = data_pl_copy / np.max(data_pl.reshape(-1, 1))
    data_pl_copy = data_pl_copy.reshape(-1, 1)
    diff_pl = data_pl_copy - model_data_pl
    diff_pl[np.abs(data_pl_copy) < 3e-2] = 0
    loglike = -0.5 * np.dot(diff_pl.T, np.dot(inv_co_var_mat_pl, diff_pl))
    Chi_squared = -2 * loglike / (len(data_pl) - len(theta))
    return loglike, Chi_squared, EX_kr, Ex_knr


def log_probability_pl(
    theta,
    data_pl,
    inv_co_var_mat_pl,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf, None, None, None, None
    log_like, Chi_squared, EX_kr, Ex_knr = pl_loglike(
        theta,
        data_pl,
        inv_co_var_mat_pl,
        X["temperature_list_pl"],
        X["hws_pl"],
        fixed_parameters_dict,
        params_to_fit,
    )
    log_prob = lp + log_like[0][0]
    if np.isnan(log_like) or np.isinf(log_like) or log_like is None:
        return -np.inf, None, None, None, None
    return log_prob, log_like[0], Chi_squared[0][0], EX_kr[-1], Ex_knr[0][-1]
