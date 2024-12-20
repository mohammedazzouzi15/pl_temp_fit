### script for plotting data from the sampling outputs

import numpy as np
from pl_temp_fit import (
    Exp_data_utils,
    covariance_utils,
    fit_pl_utils,
    config_utils,
)
import emcee
from pathlib import Path


def main(databse_path, test_id):
    add_for_ssh = "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/"  # ""
    model_config, model_config_save = config_utils.load_model_config(
        test_id, database_folder=databse_path
    )

    filename = add_for_ssh + model_config_save["save_folder"] + "/sampler.h5"
    reader = emcee.backends.HDFBackend(filename, name="multi_core")
    # distribution = reader.get_chain(discard=0, flat=True)
    plot_fit_to_experimental_data(
        model_config_save, model_config, reader, discard=5000
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
    log_likihood = reader.get_log_prob(discard=discard)
    print("log_likelihood", log_likihood)
    chi_score = -2 * log_likihood/(len(temperature_list) * len(hws) + len(temperature_list) +1 - len(params_to_fit_init))
    print("chi_score", np.mean(chi_score))
    distribution = distribution.reshape(-1, distribution.shape[-1])
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

    fig.suptitle(model_config_save["csv_name_pl"].split("/")[-1] + ", chi_score: " + str(np.mean(chi_score)), fontsize=16)
    # delete legend from ax
    for axis in ax:
        axis.get_legend().remove()
    fig.savefig("script/fit_to_data.png")

    return fig, ax


if __name__ == "__main__":
    databse_path = Path(
        "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/fit_experimental_emcee_pl/fit_data_base/allLifetimes/"
    )
    test_id = "9f97c9a5-76ce-4db3-a03a-ae4211055032"
    main(databse_path, test_id)
