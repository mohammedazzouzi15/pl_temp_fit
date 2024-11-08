# get the variance of the data and plot it
import matplotlib.pyplot as plt
import numpy as np

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


def get_covariance_matrix(
    true_model_pl_list, numnber_of_samples, model_config
):
    variance = (
        np.var(np.array(true_model_pl_list), axis=0)
        + model_config["noise_sigma"]
    )
    co_var_mat_test = np.cov(
        np.array(true_model_pl_list).reshape(numnber_of_samples, -1),
        rowvar=False,
    )
    np.fill_diagonal(
        co_var_mat_test,
        co_var_mat_test.diagonal() + model_config["noise_sigma"],
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
    """Plot the mean and variance of the data

    Arguments:
    ---------
        ax {matplotlib axis} -- axis to plot the data
        co_var_mat_test {np.array} -- covariance matrix of the data
        variance {np.array} -- variance of the data
        hws {np.array} -- photon energies
        temperature_list {np.array} -- temperatures
        true_model_pl_list {np.array} -- true model data

    Returns:
    -------
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
    """Plot the generated data

    Arguments:
    ---------
        temperature_list_el {np.array} -- temperatures for EL
        hws_el {np.array} -- photon energies for EL
        temperature_list_pl {np.array} -- temperatures for PL
        hws_pl {np.array} -- photon energies for PL
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
        model_data_pl, model_data_el = generate_data_utils.generate_data(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_pl,
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
        )
        true_model_el_list = add_data_list(
            true_model_el_list,
            model_data_el,
            model_config["hws_el"],
            model_config["temperature_list_el"],
        )

        add_plot_to_ax(
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
            ax[0][0],
            model_data_pl,
        )
        add_plot_to_ax(
            model_config["hws_el"],
            model_config["temperature_list_el"],
            ax[0][1],
            model_data_el,
        )

    co_var_mat_test_pl, variance_pl = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_el, variance_el = get_covariance_matrix(
        true_model_el_list, numnber_of_samples, model_config
    )

    co_var_mat_test_pl = plot_mean_and_variance(
        ax[1][0],
        co_var_mat_test_pl,
        variance_pl,
        model_config["hws_pl"],
        model_config["temperature_list_pl"],
        true_model_pl_list,
    )
    co_var_mat_test_el = plot_mean_and_variance(
        ax[1][1],
        co_var_mat_test_el,
        variance_el,
        model_config["hws_el"],
        model_config["temperature_list_el"],
        true_model_el_list,
    )
    # plot the generated data
    if savefig:
        fig.savefig(save_folder + "/generated_data.png")
    fig.tight_layout()
    return co_var_mat_test_pl, co_var_mat_test_el, variance_el, variance_pl


def get_covariance_matrix_for_data(
    model_config, params_to_fit, fixed_parameters_dict, numnber_of_samples=20
):
    true_model_pl_list = []
    true_model_el_list = []
    for x in range(numnber_of_samples):
        model_data_pl, model_data_el = generate_data_utils.generate_data(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_pl,
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
        )
        true_model_el_list = add_data_list(
            true_model_el_list,
            model_data_el,
            model_config["hws_el"],
            model_config["temperature_list_el"],
        )

    co_var_mat_test_pl, variance_pl = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_el, variance_el = get_covariance_matrix(
        true_model_el_list, numnber_of_samples, model_config
    )
    return co_var_mat_test_pl, co_var_mat_test_el, variance_el, variance_pl


def plot_generated_data_pl(
    save_folder,
    model_config,
    savefig=True,
    params_to_fit={},
    fixed_parameters_dict={},
    numnber_of_samples=20,
):
    """Plot the generated data

    Args:
    ----
        temperature_list_el {np.array} -- temperatures for EL
        hws_el {np.array} -- photon energies for EL
        temperature_list_pl {np.array} -- temperatures for PL
        hws_pl {np.array} -- photon energies for PL
        save_folder {str} -- folder to save the data
        model_config {dict} -- model configuration
        params_to_fit {dict} -- parameters to fit in the model
        fixed_parameters_dict {dict} -- fixed parameters for the model
    Keyword Arguments:
        savefig {bool} -- save the figure (default: {True})
        true_parameters {dict} -- true parameters for the model (default: {None})

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    true_model_pl_list = []
    true_model_el_list = []

    for x in range(numnber_of_samples):
        model_data_pl, EX_kr, Ex_knr = generate_data_utils.generate_data_pl(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_pl,
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
        )

        add_plot_to_ax(
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
            ax[0],
            model_data_pl,
        )

    co_var_mat_test_pl, variance_pl = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )
    co_var_mat_test_pl = plot_mean_and_variance(
        ax[1],
        co_var_mat_test_pl,
        variance_pl,
        model_config["hws_pl"],
        model_config["temperature_list_pl"],
        true_model_pl_list,
    )
    # plot the generated data
    if savefig:
        import os

        os.makedirs(save_folder, exist_ok=True)
        fig.savefig(save_folder + "/generated_data.png")
    fig.tight_layout()
    return co_var_mat_test_pl, variance_pl


def get_covariance_matrix_for_data_pl(
    model_config, params_to_fit, fixed_parameters_dict, numnber_of_samples=20
):
    true_model_pl_list = []
    for x in range(numnber_of_samples):
        model_data_pl, EX_kr, Ex_knr = generate_data_utils.generate_data_pl(
            **model_config,
            params_to_fit=params_to_fit,
            fixed_parameters_dict=fixed_parameters_dict,
        )
        true_model_pl_list = add_data_list(
            true_model_pl_list,
            model_data_pl,
            model_config["hws_pl"],
            model_config["temperature_list_pl"],
        )

    co_var_mat_test_pl, variance_pl = get_covariance_matrix(
        true_model_pl_list, numnber_of_samples, model_config
    )

    return co_var_mat_test_pl, variance_pl
