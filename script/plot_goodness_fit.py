### script for plotting data from the sampling outputs

import numpy as np
from pl_temp_fit import (
    Exp_data_utils,
    covariance_utils,
    fit_pl_utils,
    config_utils,
)
import emcee


def main(databse_path, test_id):
    add_for_ssh = "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/"  # ""
    model_config, model_config_save = config_utils.load_model_config(
        test_id, database_folder=databse_path
    )

    filename = add_for_ssh + model_config_save["save_folder"] + "/sampler.h5"
    reader = emcee.backends.HDFBackend(filename, name="multi_core")
    # distribution = reader.get_chain(discard=0, flat=True)
    plot_fit_to_experimental_data(
        model_config_save, model_config, reader, discard=20
    )


def plot_fit_to_experimental_data(
    model_config_save,
    model_config,
    reader,
    discard=10,
    chains_list=None,
    filter_log_likelihood="",
):
    """Plot the fit to the experimental data
    model_config_save: the model config save dictionary
    model_config: the model config dictionary
    reader: the reader object from the emcee sampler
    discard: the number of samples to discard
    chains_list: the list of chains to plot
    """

    save_folder = model_config_save["save_folder"]
    fixed_parameters_dict = model_config_save["fixed_parameters_dict"]
    params_to_fit_init = model_config_save["params_to_fit_init"]
    csv_name = model_config_save["csv_name_pl"]
    
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    distribution = reader.get_chain(discard=discard)
    if chains_list is not None:
        distribution = distribution[:, chains_list, :].reshape(
            -1, distribution.shape[-1]
        )
    else:
        distribution = distribution.reshape(-1, distribution.shape[-1])
        distribution = eval(f"distribution[{filter_log_likelihood}]")
    true_parameters = fit_pl_utils.get_param_dict(
        params_to_fit_init, distribution[-1]
    )  # model_config_save['params_to_fit_init']#
    _, variance_pl = covariance_utils.get_covariance_matrix_for_data_pl(
        model_config,
        true_parameters,
        fixed_parameters_dict,
        numnber_of_samples=20,
    )
    fig, ax = fit_pl_utils.plot_exp_data_with_variance(
        temperature_list,
        hws,
        variance_pl,
        save_folder,
        fixed_parameters_dict,
        true_parameters,
        Exp_data,
    )
    for true_parameters in distribution[
        np.random.choice(len(distribution), 10), :
    ]:
        true_parameters = fit_pl_utils.get_param_dict(
            params_to_fit_init, true_parameters
        )

        fig, ax = fit_pl_utils.plot_exp_data_with_variance(
            temperature_list,
            hws,
            variance_pl,
            save_folder,
            fixed_parameters_dict,
            true_parameters,
            Exp_data,
            fig=fig,
            axis=ax,
        )

    # delete legend from ax
    for axis in ax:
        axis.get_legend().remove()

    return fig, ax
