import os
from pathlib import Path

import pandas as pd

from pl_temp_fit import Exp_data_utils, config_utils

os.chdir(Path(__file__).parent)
os.makedirs("figures", exist_ok=True)
os.makedirs("fit_data_base", exist_ok=True)
os.makedirs("fit_data", exist_ok=True)


def load_and_plot_data(csv_name="example_PL_data_Y6.csv"):
    # Load and plot the experimental data

    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    fig, ax = Exp_data_utils.plot_pl_data(
        Exp_data, temperature_list, hws, title="Experimental Data"
    )
    fig.savefig("figures/example_PL_data_Y6.png")
    return Exp_data, temperature_list, hws


def modify_exp_data(csv_name="example_PL_data_Y6.csv"):
    # change the wavelength range and plot the data
    csv_names = Exp_data_utils.change_wavelength_range(
        Path(csv_name),
        hws_limits=[1.3, 1.7],
        step=0.01,
        temperature_split=[],
    )
    for xsc_name in csv_names:
        Exp_data, temperature_list, hws = Exp_data_utils.read_data(xsc_name)
        fig, ax = Exp_data_utils.plot_pl_data(
            Exp_data, temperature_list, hws, title=xsc_name.name.split("/")[-1]
        )
        print(xsc_name)


def read_and_plot_lifetime_data(csv_name="temperature_lifetimes_exp.csv"):
    # Read and plot the lifetime data
    df_lifetime = pd.read_csv(csv_name)
    temperature_lifetimes_exp = {
        f"{row['Temperature (K)']} K": row["Lifetime (s)"]
        for _, row in df_lifetime.iterrows()
    }
    max_abs_pos_exp = df_lifetime["Max Abs Pos Exp"].max()
    return temperature_lifetimes_exp, max_abs_pos_exp


def generate_model_parameters(
    csv_name, temperature_list, hws, temperature_lifetimes_exp, max_abs_pos_exp
):
    # Define the error in the experimental data

    (
        number_free_parameters,
        Temp_std_err,
        hws_std_err,
        relative_intensity_std_error_pl,
        noise_sigma,
    ) = (5, 10, 0.005, 0.05, 0.001)
    error_in_max_abs_pos = 0.01
    relative_error_lifetime: 0.05

    # Define the model parameters

    fixed_parameters_dict = {
        "EX": {"numbrstates": 20, "disorder_ext": 0.1},
        "CT": {"off": 1},
        "D": {},
    }
    params_to_fit_init = {
        "EX": {"E": 1.7, "sigma": 0.04, "Li": 7.8e-2, "Lo": 0.11, "hO": 0.159},
        "CT": {},
        "D": {},
    }
    min_bounds = {
        "EX": {"E": 1.5, "sigma": 0.001, "Li": 0.03, "Lo": 0.03, "hO": 0.1},
        "CT": {},
        "D": {},
    }
    max_bounds = {
        "EX": {"E": 1.9, "sigma": 0.1, "Li": 0.2, "Lo": 0.2, "hO": 0.2},
        "CT": {},
        "D": {},
    }

    # save the model config

    model_config, test_id = config_utils.save_model_config(
        csv_name_pl=Path(csv_name),
        Temp_std_err=Temp_std_err,
        hws_std_err=hws_std_err,
        relative_intensity_std_error_pl=relative_intensity_std_error_pl,
        temperature_list_pl=temperature_list,
        hws_pl=hws,
        noise_sigma=noise_sigma,
        fixed_parameters_dict=fixed_parameters_dict,
        params_to_fit_init=params_to_fit_init,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        num_iteration_max_likelihood=5,
        coeff_spread=0.5,
        nsteps=10,
        num_coords=32,
        database_folder="fit_data_base/",
        data_folder="fit_data/",
        temperature_lifetimes_exp=temperature_lifetimes_exp,
        max_abs_pos_exp=max_abs_pos_exp,
    )


if __name__ == "__main__":
    # Load and plot the experimental data
    Exp_data, temperature_list, hws = load_and_plot_data()

    # Modify the experimental data
    modify_exp_data()

    # Read and plot the lifetime data
    temperature_lifetimes_exp, max_abs_pos_exp = read_and_plot_lifetime_data()

    # Generate model parameters and save the configuration
    generate_model_parameters(
        "example_data_Y6.csv",
        temperature_list,
        hws,
        temperature_lifetimes_exp,
        max_abs_pos_exp,
    )
