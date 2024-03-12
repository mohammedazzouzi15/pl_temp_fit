# define a pytensor Op for our likelihood function
from pl_temp_fit import Exp_data_utils
import numpy as np
from pl_temp_fit import config_utils
from pl_temp_fit import (
    covariance_utils,
    fit_EL_utils,
)


def main(model_config_id):
    """
    Run the EL sample fitting
    """
    model_config, model_config_save = config_utils.load_model_config(
        model_config_id
    )
    csv_name_EL = model_config_save["csv_name_EL"]
    csv_name_PL = model_config_save["csv_name_PL"]
    # Load the data
    Exp_data_PL, temperature_list_PL, hws_PL = Exp_data_utils.read_data(
        csv_name_PL
    )
    Exp_data_EL, temperature_list_EL, hws_EL = Exp_data_utils.read_data(
        csv_name_EL
    )
    # Load the model config
    model_config, model_config_save = config_utils.load_model_config(
        model_config_id
    )
    fixed_parameters_dict, params_to_fit_init, min_bounds, max_bounds = (
        config_utils.get_dict_params(model_config_save)
    )
    model_config["temperature_list_EL"] = temperature_list_EL
    model_config["hws_EL"] = hws_EL
    model_config["temperature_list_PL"] = temperature_list_PL
    model_config["hws_PL"] = hws_PL
    save_folder = model_config_save["save_folder"]

    co_var_mat_PL, co_var_mat_EL, variance_EL, variance_PL = (
        covariance_utils.get_covariance_matrix_for_data(
            model_config,
            fixed_parameters_dict=fixed_parameters_dict,
            params_to_fit=params_to_fit_init,
        )
    )

    soln = fit_EL_utils.get_maximum_likelihood_estimate(
        Exp_data_EL,
        Exp_data_PL,
        co_var_mat_PL,
        co_var_mat_EL,
        model_config,
        save_folder,
        num_coords=model_config_save['num_iteration_max_likelihood'],
        fixed_parameters_dict=fixed_parameters_dict,
        params_to_fit=params_to_fit_init,
        min_bound=min_bounds,
        max_bound=max_bounds,
    )
    true_parameters = fit_EL_utils.get_param_dict(params_to_fit_init, soln.x)
    co_var_mat_PL, co_var_mat_EL, variance_EL, variance_PL = (
        covariance_utils.get_covariance_matrix_for_data(
            model_config,
            fixed_parameters_dict=fixed_parameters_dict,
            params_to_fit=true_parameters,
        )
    )

    fit_EL_utils.run_sampler_parallel(
        save_folder,
        Exp_data_EL,
        Exp_data_PL,
        co_var_mat_EL,
        co_var_mat_PL,
        true_parameters,
        fixed_parameters_dict,
        min_bounds,
        max_bounds,
        model_config,
        nsteps=model_config_save['nsteps'],
        coeff_spread=model_config_save['coeff_spread'],
        num_coords=model_config_save['num_coords'],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the EL sample fitting")
    parser.add_argument(
        "--model_config_id",
        type=str,
        help="The id of the model config to use",
    )
    args = parser.parse_args()
    main(args.model_config_id)
