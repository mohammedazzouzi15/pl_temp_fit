"""This module contains functions to save and load the model configuration and the data used for the fit.

The model configuration is saved in a json file and the data is saved in a csv file.
The model configuration is used to generate the data and the data is used to fit the model.
"""

import datetime
import json
import os
import uuid
from pathlib import Path

from pl_temp_fit import Exp_data_utils


def save_model_config(
    csv_name_pl="",
    csv_name_el="",
    Temp_std_err=0.1,
    hws_std_err=0.005,
    relative_intensity_std_error_pl=0.001,
    relative_intensity_std_error_el=0.001,
    temperature_list_pl=[],
    hws_pl=[],
    temperature_list_el=[],
    hws_el=[],
    noise_sigma=0.01,
    fixed_parameters_dict={},
    params_to_fit_init={},
    min_bounds={},
    max_bounds={},
    num_iteration_max_likelihood=5,
    coeff_spread=0.5,
    nsteps=10000,
    num_coords=32,
    database_folder="fit_experimental_emcee_el/fit_data_base/",
    data_folder="fit_experimental_emcee_el/fit_data/",
    test_id="",
    max_abs_pos_exp=0.5,
    error_in_max_abs_pos=0.1,
    temperature_lifetimes_exp={},
    relative_error_lifetime=0.05,
):
    """Save the model configuration and the data used for the fit.

    Args:
    ----
        csv_name_pl (str): Name of the PL CSV file.
        csv_name_el (str): Name of the EL CSV file.
        Temp_std_err (float): Standard error for temperature.
        hws_std_err (float): Standard error for hws.
        relative_intensity_std_error_pl (float): Relative intensity standard error for PL.
        relative_intensity_std_error_el (float): Relative intensity standard error for EL.
        temperature_list_pl (list): List of temperatures for PL.
        hws_pl (list): List of hws for PL.
        temperature_list_el (list): List of temperatures for EL.
        hws_el (list): List of hws for EL.
        noise_sigma (float): Noise sigma value.
        fixed_parameters_dict (dict): Dictionary of fixed parameters.
        params_to_fit_init (dict): Initial parameters to fit.
        min_bounds (dict): Minimum bounds for parameters.
        max_bounds (dict): Maximum bounds for parameters.
        num_iteration_max_likelihood (int): Number of iterations for maximum likelihood.
        coeff_spread (float): Coefficient spread value.
        nsteps (int): Number of steps.
        num_coords (int): Number of coordinates.
        database_folder (str): Folder to save the database.
        data_folder (str): Folder to save the data.
        test_id (str): Test identifier.
        max_abs_pos_exp (float): Maximum absolute position for experiment.
        error_in_max_abs_pos (float): Error in maximum absolute position.
        temperature_lifetimes_exp (dict): Dictionary of temperature lifetimes for experiment.
        relative_error_lifetime (float): Relative error in lifetime.

    Returns:
    -------
        tuple: A tuple containing the model configuration dictionary and the test ID.

    """
    model_config = {
        "Temp_std_err": Temp_std_err,
        "hws_std_err": hws_std_err,
        "relative_intensity_std_error_pl": relative_intensity_std_error_pl,
        "relative_intensity_std_error_el": relative_intensity_std_error_el,
        "temperature_list_pl": temperature_list_pl,
        "hws_pl": hws_pl,
        "temperature_list_el": temperature_list_el,
        "hws_el": hws_el,
        "noise_sigma": noise_sigma,
    }
    print(f"size of hw is {hws_pl.shape}")
    print(f"size of temperature_list is {temperature_list_pl.shape}")
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    test_id = test_id or str(uuid.uuid4())

    save_folder = Path(data_folder, csv_name_pl.name.split(".")[0], test_id)
    save_folder.mkdir(parents=True, exist_ok=True)

    model_config_save = {
        **model_config,
        "save_folder": save_folder.absolute().as_posix(),
        "csv_name_pl": csv_name_pl.absolute().as_posix(),
        "csv_name_el": csv_name_el.absolute().as_posix()
        if csv_name_el
        else "",
        "date": date,
        "test_id": test_id,
        "fixed_parameters_dict": fixed_parameters_dict,
        "params_to_fit_init": params_to_fit_init,
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "num_iteration_max_likelihood": num_iteration_max_likelihood,
        "coeff_spread": coeff_spread,
        "nsteps": nsteps,
        "num_coords": num_coords,
        "max_abs_pos_exp": max_abs_pos_exp,
        "error_in_max_abs_pos": error_in_max_abs_pos,
        "temperature_lifetimes_exp": temperature_lifetimes_exp,
        "relative_error_lifetime": relative_error_lifetime,
    }

    for key in [
        "temperature_list_pl",
        "hws_pl",
        "temperature_list_el",
        "hws_el",
    ]:
        model_config_save.pop(key)

    os.makedirs(database_folder, exist_ok=True)
    with open(Path(database_folder, f"{test_id}.json"), "w") as f:
        json.dump(model_config_save, f, indent=4)

    return model_config, test_id


def updata_model_config(
    test_id,
    database_folder: Path,
    model_config_save,
):
    """Update the model configuration in the database.

    Args:
    ----
        test_id (str): Test identifier.
        database_folder (Path): Path to the database folder.
        model_config_save (dict): Model configuration dictionary to save.

    Returns:
    -------
        str: The test ID.

    """
    with Path(database_folder, f"{test_id}.json").open("w") as f:
        json.dump(model_config_save, f, indent=4)

    return test_id


def load_model_config(
    test_id,
    database_folder: Path,
):
    """Load the model configuration from the database.

    Args:
    ----
        test_id (str): Test identifier.
        database_folder (Path): Path to the database folder.

    Returns:
    -------
        tuple: A tuple containing the model configuration dictionary and the saved model configuration dictionary.

    """
    with Path(database_folder, f"{test_id}.json").open("r") as f:
        model_config_save = json.load(f)

    model_config = {
        "Temp_std_err": 0,
        "hws_std_err": 0,
        "relative_intensity_std_error_pl": 0,
        "relative_intensity_std_error_el": 0,
        "noise_sigma": 0,
    }

    for keys in model_config:
        if keys in model_config_save:
            model_config[keys] = model_config_save[keys]

    import os

    if "csv_name_pl" in model_config_save:
        csv_name = model_config_save["csv_name_pl"]
        if os.path.exists(csv_name):
            Exp_data, temperature_list_pl, hws_pl = Exp_data_utils.read_data(
                csv_name
            )
            model_config["temperature_list_pl"] = temperature_list_pl
            model_config["hws_pl"] = hws_pl
        else:
            model_config["temperature_list_pl"] = []
            model_config["hws_pl"] = []
    if "csv_name_el" in model_config_save:
        csv_name = model_config_save["csv_name_el"]
        if os.path.exists(csv_name):
            Exp_data, temperature_list_el, hws_el = Exp_data_utils.read_data(
                csv_name
            )
            model_config["temperature_list_el"] = temperature_list_el
            model_config["hws_el"] = hws_el
        else:
            model_config["temperature_list_el"] = []
            model_config["hws_el"] = []

    return model_config, model_config_save


def get_dict_params(model_config):
    """Get the dictionary parameters from the model configuration.

    Args:
    ----
        model_config (dict): Model configuration dictionary.

    Returns:
    -------
        tuple: A tuple containing fixed parameters dictionary, parameters to fit, minimum bounds, and maximum bounds.

    """
    fixed_parameters_dict = model_config["fixed_parameters_dict"]
    params_to_fit = model_config["params_to_fit_init"]
    min_bound = model_config["min_bounds"]
    max_bound = model_config["max_bounds"]
    return fixed_parameters_dict, params_to_fit, min_bound, max_bound


def update_csv_name_pl(
    model_config_save,
    csv_name,
    new_path,
    new_path_save_folder,
    script_to_run,
    test_id,
):
    """Updates the 'csv_name_pl' and 'save_folder' fields in the model configuration dictionary.

    Args:
    ----
        model_config_save (dict): The model configuration dictionary to be updated.
        csv_name (str): The original CSV file name with its path.
        new_path (str): The new path where the CSV file will be saved.
        new_path_save_folder (str): The new path for the save folder.
        script_to_run (str): The name of the script to be run.
        test_id (str): The test identifier.

    Returns:
    -------
        dict: The updated model configuration dictionary with new 'csv_name_pl' and 'save_folder' fields.

    """
    model_config_save["csv_name_pl"] = new_path + "/" + csv_name.split("/")[-1]
    model_config_save["save_folder"] = (
        new_path_save_folder
        + "/"
        + csv_name.split("/")[-1].replace(".csv", "")
        + "/"
        + script_to_run
        + "/"
        + test_id
    )
    return model_config_save
