# get the variance of the data and plot it
import numpy as np
import matplotlib.pyplot as plt
from pl_temp_fit import generate_data_utils


def add_data_list(true_model_list, model_data, hws, temperature_list):
    data_true_plot = model_data.reshape(len(hws), -1)
    data_true_plot = data_true_plot / max(data_true_plot.reshape(-1, 1))
    true_model_list.append(
        data_true_plot.reshape(len(hws), len(temperature_list))
        + np.random.normal(0, 0.01, size=(len(hws), len(temperature_list)))
    )
    return true_model_list


def add_plot_to_ax(hws, temperature_list, ax, model_data):
    data_true_plot = model_data.reshape(len(hws), -1)
    data_true_plot = data_true_plot / max(data_true_plot.reshape(-1, 1))
    for i in range(len(temperature_list)):
        ax.plot(
            hws,
            data_true_plot[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
            alpha=0.3,
        )
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel(" Intensity (arb. units)")
    ax.set_title("Posterior mean prediction")


def get_covariance_matrix(true_model_pl_list, numnber_of_samples, model_config):
    variance = np.var(np.array(true_model_pl_list), axis=0)+ model_config['sigma']
    co_var_mat_test = np.cov(
        np.array(true_model_pl_list).reshape(numnber_of_samples, -1),
        rowvar=False,
    )
    np.fill_diagonal(
        co_var_mat_test, co_var_mat_test.diagonal() +model_config['sigma']
    )
    return co_var_mat_test, variance


def plot_mean_and_variance(
    ax,
    co_var_mat_test,
    variance,
    hws,
    temperature_list,
    true_model_pl_list,
):
    """plot the mean and variance of the data

    Arguments:
        ax {matplotlib axis} -- axis to plot the data
        co_var_mat_test {np.array} -- covariance matrix of the data
        variance {np.array} -- variance of the data
        hws {np.array} -- photon energies
        temperature_list {np.array} -- temperatures
        true_model_pl_list {np.array} -- true model data

    Returns:
        np.array -- covariance matrix of the data
    """
    mean_value_plot = np.mean(np.array(true_model_pl_list), axis=0)
    print(f"shape of mean value plot is {mean_value_plot.shape}")
    # plot the generated data
    for i in range(len(temperature_list)):

        ax.plot(
            hws,
            mean_value_plot[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
        )
        ax.fill_between(
            hws,
            np.mean(np.array(true_model_pl_list), axis=0)[:, i]
            - np.sqrt(variance[:, i]),
            np.mean(np.array(true_model_pl_list), axis=0)[:, i]
            + np.sqrt(variance[:, i]),
            alpha=0.3,
            color="C" + str(i),
        )
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL Intensity (arb. units)")
        ax.set_title("mean prediction with 1 std dev")
    return co_var_mat_test


def plot_generated_data(
    save_folder,
    model_config,
    savefig=True,
    params_to_fit={},
    fixed_parameters_dict={},
    numnber_of_samples=20,
):
    """
    plot the generated data

    Arguments:
        temperature_list_EL {np.array} -- temperatures for EL
        hws_EL {np.array} -- photon energies for EL
        temperature_list_PL {np.array} -- temperatures for PL
        hws_PL {np.array} -- photon energies for PL
        save_folder {str} -- folder to save the data
        model_config {dict} -- model configuration
        params_to_fit {dict} -- parameters to fit in the model
        fixed_parameters_dict {dict} -- fixed parameters for the model
    Keyword Arguments:
        savefig {bool} -- save the figure (default: {True})
        true_parameters {dict} -- true parameters for the model (default: {None})
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    true_model_pl_list = []
    true_model_el_list = []

    for x in range(numnber_of_samples):
        model_data_PL, model_data_EL = generate_data_utils.generate_data(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_PL,
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
        )
        true_model_el_list = add_data_list(
            true_model_el_list,
            model_data_EL,
            model_config["hws_EL"],
            model_config["temperature_list_EL"],
        )

        add_plot_to_ax(
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
            ax[0][0],
            model_data_PL,
        )
        add_plot_to_ax(
            model_config["hws_EL"],
            model_config["temperature_list_EL"],
            ax[0][1],
            model_data_EL,
        )

    co_var_mat_test_PL, variance_PL = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_EL, variance_EL = get_covariance_matrix(
        true_model_el_list, numnber_of_samples, model_config
    )

    co_var_mat_test_PL = plot_mean_and_variance(
        ax[1][0],
        co_var_mat_test_PL,
        variance_PL,
        model_config["hws_PL"],
        model_config["temperature_list_PL"],
        true_model_pl_list,
    )
    co_var_mat_test_EL = plot_mean_and_variance(
        ax[1][1],
        co_var_mat_test_EL,
        variance_EL,
        model_config["hws_EL"],
        model_config["temperature_list_EL"],
        true_model_el_list,
    )
    # plot the generated data
    if savefig:
        fig.savefig(save_folder + "/generated_data.png")
    fig.tight_layout()
    return co_var_mat_test_PL, co_var_mat_test_EL, variance_EL, variance_PL


def get_covariance_matrix_for_data(
    model_config, params_to_fit, fixed_parameters_dict, numnber_of_samples=20
):
    true_model_pl_list = []
    true_model_el_list = []
    for x in range(numnber_of_samples):
        model_data_PL, model_data_EL = generate_data_utils.generate_data(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_PL,
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
        )
        true_model_el_list = add_data_list(
            true_model_el_list,
            model_data_EL,
            model_config["hws_EL"],
            model_config["temperature_list_EL"],
        )

    co_var_mat_test_PL, variance_PL = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_EL, variance_EL = get_covariance_matrix(
        true_model_el_list, numnber_of_samples, model_config
    )
    return co_var_mat_test_PL, co_var_mat_test_EL, variance_EL, variance_PL


def plot_generated_data_PL(
    save_folder,
    model_config,
    savefig=True,
    params_to_fit={},
    fixed_parameters_dict={},
    numnber_of_samples=20,
):
    """
    plot the generated data

    Arguments:
        temperature_list_EL {np.array} -- temperatures for EL
        hws_EL {np.array} -- photon energies for EL
        temperature_list_PL {np.array} -- temperatures for PL
        hws_PL {np.array} -- photon energies for PL
        save_folder {str} -- folder to save the data
        model_config {dict} -- model configuration
        params_to_fit {dict} -- parameters to fit in the model
        fixed_parameters_dict {dict} -- fixed parameters for the model
    Keyword Arguments:
        savefig {bool} -- save the figure (default: {True})
        true_parameters {dict} -- true parameters for the model (default: {None})
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    true_model_pl_list = []
    true_model_el_list = []

    for x in range(numnber_of_samples):
        model_data_PL,EX_kr, Ex_knr  = generate_data_utils.generate_data_PL(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_PL,
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
        )


        add_plot_to_ax(
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
            ax[0],
            model_data_PL,
        )

    co_var_mat_test_PL, variance_PL = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_PL = plot_mean_and_variance(
        ax[1],
        co_var_mat_test_PL,
        variance_PL,
        model_config["hws_PL"],
        model_config["temperature_list_PL"],
        true_model_pl_list,
    )
    # plot the generated data
    if savefig:
        fig.savefig(save_folder + "/generated_data.png")
    fig.tight_layout()
    return co_var_mat_test_PL, variance_PL



def get_covariance_matrix_for_data_PL(
    model_config, params_to_fit, fixed_parameters_dict, numnber_of_samples=20
):
    true_model_pl_list = []
    for x in range(numnber_of_samples):
        model_data_PL,EX_kr, Ex_knr  = generate_data_utils.generate_data_PL(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_PL,
            model_config["hws_PL"],
            model_config["temperature_list_PL"],
        )


    co_var_mat_test_PL, variance_PL = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )

    return co_var_mat_test_PL, variance_PL
