"""This module contains functions to save and load the model configuration.

and the data used for the fit.
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
    sigma=0.01,
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
):
    """Save the model configuration and the data used for the fit.
    
    Args:
    ----
        csv_name_pl (str): The path to the csv file containing the PL data.
        csv_name_el (str): The path to the csv file containing the EL data.
        Temp_std_err (float): The standard error of the temperature measurement.
        hws_std_err (float): The standard error of the hws measurement.
    relative_intensity_std_error_pl (float): The standard error of the PL relative intensity measurement.

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
        "sigma": sigma,
    }
    print(f"size of hw is {hws_pl.shape}")
    print(f"size of temperature_list is {temperature_list_pl.shape}")
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    if test_id == "":
        test_id = str(uuid.uuid4())
    # generate the data

    save_folder = Path(data_folder, csv_name_pl.name.split(".")[0],test_id)
    save_folder.mkdir(parents=True, exist_ok=True)
    # save _model_config

    # get initial covariance matrix
    # get covariance matrix for the experimental data
    model_config_save = model_config.copy()
    model_config_save["save_folder"] = save_folder.absolute().as_posix()
    model_config_save["csv_name_pl"] = csv_name_pl.absolute().as_posix()
    if csv_name_el == "":
        model_config_save["csv_name_el"] = ""
    else:
        model_config_save["csv_name_el"] = csv_name_el.absolute().as_posix()
    model_config_save["date"] = date
    model_config_save["test_id"] = test_id
    model_config_save["fixed_parameters_dict"] = fixed_parameters_dict
    model_config_save["params_to_fit_init"] = params_to_fit_init
    model_config_save["min_bounds"] = min_bounds
    model_config_save["max_bounds"] = max_bounds
    model_config_save["num_iteration_max_likelihood"] = (
        num_iteration_max_likelihood
    )
    model_config_save["coeff_spread"] = coeff_spread
    model_config_save["nsteps"] = nsteps
    model_config_save["num_coords"] = num_coords

    model_config_save.pop("temperature_list_pl")
    model_config_save.pop("hws_pl")
    model_config_save.pop("temperature_list_el")
    model_config_save.pop("hws_el")

    os.makedirs(database_folder, exist_ok=True)
    with open(database_folder + f"/{test_id}.json", "w") as f:
        json.dump(model_config_save, f)

    return model_config, test_id


def load_model_config(
    test_id,
    database_folder: Path,
):
    with Path(database_folder,f"{test_id}.json").open("r") as f:
        model_config_save = json.load(f)

    model_config = {
        "Temp_std_err": 0,
        "hws_std_err": 0,
        "relative_intensity_std_error_pl": 0,
        "relative_intensity_std_error_el": 0,
        "sigma": 0,
    }

    for keys in model_config:
        model_config[keys] = model_config_save[keys]
    import os

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
    fixed_parameters_dict = model_config["fixed_parameters_dict"]
    params_to_fit = model_config["params_to_fit_init"]
    min_bound = model_config["min_bounds"]
    max_bound = model_config["max_bounds"]
    return fixed_parameters_dict, params_to_fit, min_bound, max_bound
