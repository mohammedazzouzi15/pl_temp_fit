import logging
from pathlib import Path

from pl_temp_fit import (
    Exp_data_utils,
    config_utils,
    fit_pl_utils,
)
from pl_temp_fit.data_generators import PLAbsAndLifetime60K


def main(model_config_id):
    model_config, model_config_save = config_utils.load_model_config(
        model_config_id,
        database_folder="fit_experimental_emcee_pl/fit_data_base/",
    )
    csv_name_pl = model_config_save["csv_name_pl"]
    save_folder = model_config_save["save_folder"]
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    # Load the data
    Exp_data_pl, temperature_list_pl, hws_pl = Exp_data_utils.read_data(
        csv_name_pl
    )
    # initialising the data generator
    pl_data_gen = PLAbsAndLifetime60K.PLAbsAndLifetime60K(
        temperature_list_pl, hws_pl
    )
    pl_data_gen.error_in_lifetime_high_temp = model_config_save[
        "relative_error_lifetime"
    ]
    pl_data_gen.error_in_max_abs_pos = model_config_save[
        "error_in_max_abs_pos"
    ]
    pl_data_gen.max_abs_pos_exp = model_config_save["max_abs_pos_exp"]

    pl_data_gen.update_with_model_config(model_config_save)
    pl_data_gen.lifetime_exp_high_temp = model_config_save[
        "temperature_lifetimes_exp"
    ]["60"]
    co_var_mat_pl, variance_pl = pl_data_gen.get_covariance_matrix()
    # getting the maximum likelihood estimate
    get_maximum_likelihood_estimate = False
    if get_maximum_likelihood_estimate:
        logging.info("Getting maximum likelihood estimate")
        soln_min = pl_data_gen.get_maximum_likelihood_estimate(
            Exp_data_pl,
            co_var_mat_pl,
            save_folder,
            coeff_spread=0.1,
            num_coords=32,
        )

        pl_data_gen.params_to_fit_init = fit_pl_utils.get_param_dict(
            pl_data_gen.params_to_fit_init, soln_min.x
        )
        co_var_mat_pl, variance_pl = pl_data_gen.get_covariance_matrix()
    logging.info("Running sampler")
    fit_pl_utils.run_sampler_parallel(
        save_folder,
        Exp_data_pl,
        co_var_mat_pl,
        pl_data_gen,
        nsteps=model_config_save["nsteps"],
        coeff_spread=model_config_save["coeff_spread"],
        num_coords=model_config_save["num_coords"],
        restart_sampling=True,
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
