### script for plotting data from the sampling outputs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pl_temp_fit import fit_PL_utils
from pl_temp_fit import covariance_utils
from pl_temp_fit import Exp_data_utils


def plot_fit_statistics(
    reader, range_chi_square=(0, 3), range_log_prior=(-1000, 0), discard=5
):
    """plot the fit statistics from the sampling output
    reader: the reader object from the emcee sampler
    range_chi_square: the range for the chi square plot
    range_log_prior: the range for the log prior plot
    discard: the number of samples to discard
    """

    print("number of iterations", reader.iteration)
    blobs = reader.get_blobs(flat=True, discard=discard)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].hist(
        blobs["Chi square"],
        30,
        color="C" + str(0),
        linewidth=2,
        histtype="step",
        range=range_chi_square,
    )
    ax[0].set_xlabel("Chi square")
    ax[0].set_ylabel("Number of samples")
    ax[0].set_title("Chi square distribution")
    ax[1].hist(
        blobs["log_likelihood"],
        30,
        color="C" + str(1),
        linewidth=2,
        histtype="step",
        range=range_log_prior,
    )
    ax[1].set_xlabel("log likelihood")
    ax[1].set_ylabel("Number of samples")
    ax[1].set_title("log likelihood distribution")
    ax[2].plot(
        blobs["log_likelihood"],
        color="C" + str(2),
        linewidth=2,
    )
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("log likelihood")
    fig.tight_layout()
    for i in range(3):
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.show()


def plot_lifetime(
    reader,
    range_chi_square=(7, 11),
    range_log_prior=(7, 11),
    discard=5,
    temperature=300,
):
    """plot the lifetime distribution from the sampling output
    reader: the reader object from the emcee sampler
    range_chi_square: the range for the chi square plot
    range_log_prior: the range for the log prior plot
    discard: the number of samples to discard
    temperature: the temperature at which the lifetime is calculated
    """
    print("number of iterations", reader.iteration)
    blobs = reader.get_blobs(flat=True, discard=discard)
    fig, ax = plt.subplots(2, 2, figsize=(7, 5))

    ax = ax.flatten()
    ax[0].hist(
        np.log10(blobs["Ex_knr"]),
        30,
        histtype="step",
        range=range_chi_square,
        color="C" + str(0),
        linewidth=2,
    )
    ax[0].set_xlabel("Log of k_nr at " + str(temperature) + " K")
    ax[0].set_ylabel("Number of samples")
    ax[1].hist(
        np.log10(blobs["Ex_kr"]),
        30,
        histtype="step",
        range=range_log_prior,
        color="C" + str(1),
        linewidth=2,
    )
    ax[1].set_xlabel("Log of k_r at " + str(temperature) + " K")
    ax[1].set_ylabel("Number of samples")
    PL_QE = blobs["Ex_kr"] / (blobs["Ex_kr"] + blobs["Ex_knr"])
    ax[2].hist(
        np.log10(PL_QE),
        30,
        histtype="step",
        range=(-4, 0),
        color="C" + str(2),
        linewidth=2,
    )
    ax[2].set_xlabel("log PL QE")
    ax[2].set_ylabel("Numer of samples")
    lifetime = 1 / (blobs["Ex_kr"] + blobs["Ex_knr"])
    ax[3].hist(
        lifetime * 1e9,
        30,
        histtype="step",
        color="C" + str(3),
        linewidth=2,
    )
    ax[3].set_xlabel("lifetime (nanoseconds)")
    ax[3].set_ylabel("Numer of samples")
    # set y ticks to exponent form
    for i in range(4):
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    plt.show()


def plot_chains(reader, model_config_save, discard=50):
    """plot the chains from the sampling output
    reader: the reader object from the emcee sampler
    model_config_save: the model config save dictionary
    discard: the number of samples to discard
    """
    csv_name = model_config_save["csv_name_PL"]
    label_list = []
    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
    labels = label_list
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
    samples = reader.get_chain(discard=discard)
    # labels = ["E", "sigma_E", "LI", "L0", "H0"]
    ndim = len(labels)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3, color="C" + str(i))
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.suptitle(f"Sampler chain for {csv_name.split('/')[-1]}")
    fig.show()


def plot_diff_chains(
    reader, model_config_save, discard=50, chains_list=[0, 1, 2, 3, 4]
):
    """plot the chains from the sampling output
    reader: the reader object from the emcee sampler
    model_config_save: the model config save dictionary
    discard: the number of samples to discard
    chains_list: the list of chains to plot
    """

    csv_name = model_config_save["csv_name_PL"]
    label_list = []
    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
    labels = label_list
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
    samples = reader.get_chain(discard=discard)
    # labels = ["E", "sigma_E", "LI", "L0", "H0"]
    ndim = len(labels)
    for i in range(ndim):
        ax = axes[i]
        for j in chains_list:

            ax.plot(
                samples[:, j, i],
                alpha=0.3,
                color="C" + str(j),
                label=f"chain {j}",
            )
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    # set legend outside the plot
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[-1].set_xlabel("step number")
    fig.suptitle(f"Sampler chain for {csv_name.split('/')[-1]}")
    fig.show()


def plot_fit_to_experimental_data(
    model_config_save, model_config, reader, discard=10, chains_list=None
):
    """plot the fit to the experimental data
    model_config_save: the model config save dictionary
    model_config: the model config dictionary
    reader: the reader object from the emcee sampler
    discard: the number of samples to discard
    chains_list: the list of chains to plot
    """
    Temp_std_err = model_config_save["Temp_std_err"]
    hws_std_err = model_config_save["hws_std_err"]
    relative_intensity_std_error_PL = model_config_save[
        "relative_intensity_std_error_PL"
    ]
    sigma = model_config_save["sigma"]
    save_folder = model_config_save["save_folder"]
    fixed_parameters_dict = model_config_save["fixed_parameters_dict"]
    params_to_fit_init = model_config_save["params_to_fit_init"]
    min_bounds = model_config_save["min_bounds"]
    max_bounds = model_config_save["max_bounds"]
    csv_name = model_config_save["csv_name_PL"]
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    distribution = reader.get_chain(discard=discard)
    if chains_list is not None:
        distribution = distribution[:, chains_list, :].reshape(
            -1, distribution.shape[-1]
        )
    else:
        distribution = distribution.reshape(-1, distribution.shape[-1])
    true_parameters = fit_PL_utils.get_param_dict(
        params_to_fit_init, distribution[-1]
    )  # model_config_save['params_to_fit_init']#
    co_var_mat_PL, variance_PL = covariance_utils.plot_generated_data_PL(
        save_folder,
        model_config,
        savefig=True,
        fixed_parameters_dict=fixed_parameters_dict,
        params_to_fit=true_parameters,
    )
    fig, ax = fit_PL_utils.plot_exp_data_with_variance(
        temperature_list,
        hws,
        variance_PL,
        save_folder,
        fixed_parameters_dict,
        true_parameters,
        Exp_data,
    )
    for true_parameters in distribution[
        np.random.choice(len(distribution), 10), :
    ]:

        true_parameters = fit_PL_utils.get_param_dict(
            params_to_fit_init, true_parameters
        )

        fig, ax = fit_PL_utils.plot_exp_data_with_variance(
            temperature_list,
            hws,
            variance_PL,
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


def plot_distribution(reader, model_config_save, discard=10):
    """plot the distribution of the parameters from the sampling output
    reader: the reader object from the emcee sampler
    model_config_save: the model config save dictionary
    discard: the number of samples to discard
    """
    csv_name = model_config_save["csv_name_PL"]
    label_list = []
    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
    labels = label_list
    ndim = len(labels)
    distribution = reader.get_chain(discard=discard, thin=5, flat=True)
    fig, axes = plt.subplots(5, figsize=(10, 7))
    axes_xlim = [[1, 2], [0, 0.03], [0, 0.2], [0, 0.2], [0.1, 0.2]]
    for i in range(ndim):
        ax = axes[i]
        ax.hist(
            distribution[:, i],
            200,
            color="C" + str(i),
            linewidth=2,
            histtype="step",
        )
        ax.set_ylabel(labels[i])
        ax.set_xlim(axes_xlim[i])
    fig.suptitle(f"Sampler distribution for {csv_name.split('/')[-1]}")
    fig.tight_layout()


def plot_corner(reader, model_config_save, discard=10):
    """plot the corner plot from the sampling output
    reader: the reader object from the emcee sampler
    model_config_save: the model config save dictionary
    discard: the number of samples to discard
    """
    csv_name = model_config_save["csv_name_PL"]
    label_list = []
    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
    labels = label_list
    samples = reader.get_chain(discard=discard, thin=15, flat=True)
    df_samples = pd.DataFrame(samples, columns=labels)
    g = sns.pairplot(df_samples, kind="hist", corner=True)
    g.fig.suptitle(f"Sampler corner plot for {csv_name.split('/')[-1]}")
