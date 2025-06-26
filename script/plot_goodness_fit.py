### script for plotting data from the sampling outputs

from pathlib import Path

import emcee
import numpy as np
from matplotlib import pyplot as plt

from pl_temp_fit import (
    Exp_data_utils,
    config_utils,
    covariance_utils,
    fit_pl_utils,
)


def main(databse_path, test_id):
    Path("script/goodness_fit").mkdir(parents=True, exist_ok=True)
    add_for_ssh = "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/"  # ""
    model_config, model_config_save = config_utils.load_model_config(
        test_id, database_folder=databse_path
    )

    filename = add_for_ssh + model_config_save["save_folder"] + "/sampler.h5"
    reader = emcee.backends.HDFBackend(filename, name="multi_core")
    discard = 5000
    chi_score = get_chi_score(
        reader,
        model_config_save,
        discard=discard,
    )
    plot_fit_to_experimental_data(
        model_config_save, model_config, reader, discard=discard
    )


def get_chi_score(reader, model_config_save, discard):
    params_to_fit_init = model_config_save["params_to_fit_init"]
    csv_name = model_config_save["csv_name_pl"]

    _, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    log_likihood = reader.get_log_prob(discard=discard)
    chi_score = (
        -2
        * log_likihood
        / (
            len(temperature_list) * len(hws)
            + len(temperature_list)
            + 1
            - len(params_to_fit_init)
        )
    )
    return chi_score


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
    # print("log_likelihood", log_likihood)
    chi_score = (
        -2
        * log_likihood
        / (
            len(temperature_list) * len(hws)
            + len(temperature_list)
            + 1
            - len(params_to_fit_init)
        )
    )
    # print("chi_score", np.mean(chi_score))
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
    real_lifetime = model_config_save["temperature_lifetimes_exp"]
    real_max_abs = float(model_config_save["max_abs_pos_exp"])
    print(real_max_abs)
    error_in_max_abs_pos = model_config_save["error_in_max_abs_pos"]
    error_in_lifetimes = model_config_save["relative_error_lifetime"]
    fig, ax, lifetime, max_abs = plot_exp_data_with_variance(
        temperature_list,
        hws,
        variance_pl,
        save_folder,
        fixed_parameters_dict,
        true_parameters,
        Exp_data,
    )
    # Create new axes for real lifetimes and max absorption
    # ax[-2].scatter(real_max_abs,1, 'o', label='Real Max Absorption',marker_size=20)
    ax[0].errorbar(
        real_max_abs,
        1,
        xerr=error_in_max_abs_pos,
        fmt="o",
        label="Real Max Absorption",
        color="red",
    )
    # add text next to the point
    ax[0].text(
        real_max_abs,
        1 + 0.05,
        "max  \n abs",
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="right",
    )
    ax_real_lifetime = fig.get_axes()[-1]
    temp_lifetime_data = [float(k) for k in real_lifetime.keys()]
    real_lifetime_values = [float(v) * 1e9 for v in real_lifetime.values()]
    # Plot real lifetimes
    ax_real_lifetime.plot(
        temp_lifetime_data,
        real_lifetime_values,
        "o-",
        label="Real Lifetime",
        color="red",
    )
    ax_real_lifetime.errorbar(
        temp_lifetime_data,
        real_lifetime_values,
        yerr=[error_in_lifetimes * v for v in real_lifetime_values],
        fmt="o",
        color="red",
    )
    ax_real_lifetime.plot(
        temperature_list,
        lifetime * 1e9,
        "x-",
        label="Simulated Lifetime",
        alpha=0.5,
        color="black",
    )
    ax_real_lifetime.set_xlabel("Temperature (K)")
    ax_real_lifetime.set_ylabel("Lifetime (ns)")
    ax_real_lifetime.legend()
    for true_parameters in distribution[
        np.random.choice(len(distribution), 10), :
    ]:
        true_parameters = fit_pl_utils.get_param_dict(
            params_to_fit_init, true_parameters
        )

        (fig, ax, lifetime, max_abs) = plot_exp_data_with_variance(
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
        ax_real_lifetime.plot(
            temperature_list, lifetime, "x-", alpha=0.5, color="black"
        )

    fig.suptitle(
        model_config_save["csv_name_pl"].split("/")[-1].split("_")[0]
        + ", chi_score: "
        + str(np.mean(chi_score)),
        fontsize=16,
    )
    name = model_config_save["csv_name_pl"].split("/")[-1].split("_")[0]
    # delete legend from ax
    for axis in ax[:-1]:
        axis.get_legend().remove()
    fig.tight_layout()
    fig.savefig(f"script/goodness_fit/{name}.png")

    return fig, ax


def plot_exp_data_with_variance(
    temperature_list_pl,
    hws_pl,
    variance_pl,
    save_folder,
    fixed_parameters_dict,
    true_parameters,
    Exp_data_pl,
    fig=None,
    axis=None,
):
    from pl_temp_fit.model_function import LTL

    def pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict={},
        params_to_fit={},
    ):
        """Run the model to generate the  PL spectra.

        Args:
        ----
        temperature_list_pl (np.array): The temperature list for the PL spectra
        hws_pl (np.array): The photon energies for the PL spectra
        fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
        params_to_fit (dict): The parameters to fit in the model

        Returns:
        -------
        tuple: The model data for the PL spectra and the radiative and non-radiative recombination rates

        """
        data = LTL.Data()
        data.update(**fixed_parameters_dict)
        data.update(**params_to_fit)
        data.D.Luminecence_exp = "PL"
        data.D.T = temperature_list_pl  # np.array([300.0, 150.0, 80.0])
        LTL.ltlcalc(data)
        pl_results = data.D.kr_hw  # .reshape(-1, 1)
        pl_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
        abs_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
        for i in range(len(temperature_list_pl)):
            pl_results_interp[:, i] = np.interp(
                hws_pl, data.D.hw, pl_results[:, i]
            )
            abs_results_interp[:, i] = np.interp(
                hws_pl, data.D.hw, data.D.alpha[:, i]
            )
        pl_results_interp = (
            pl_results_interp / pl_results_interp[pl_results_interp > 0].max()
        )
        abs_results_interp = (
            abs_results_interp / abs_results_interp.reshape(-1).max()
        )
        lifetime = 1 / (data.EX.kr + data.EX.knr[0])
        max_abs = data.D.hw[data.D.alpha[:, -1].argmax()]
        return pl_results_interp, abs_results_interp, lifetime, max_abs

    model_data_pl, abs_results_interp, lifetime, max_abs = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        true_parameters,
    )
    # print("lifetime", lifetime)
    # print("max_abs", max_abs)

    truemodel_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    if fig is None:
        fig, axis = plot_pl_data_with_variance(
            Exp_data_pl, temperature_list_pl, hws_pl, variance_pl, save_folder
        )

    # real_max_abs =

    for i, axes in enumerate(axis[:-1]):
        axes.plot(hws_pl, truemodel_pl[:, i], label="fit", color="C" + str(i))
        axes.plot(
            hws_pl,
            abs_results_interp[:, i],
            label="abs",
            color="C" + str(i),
            linestyle="--",
        )
        axes.legend()
        axes.set_ylim(0, 1.1)

    fig.suptitle("PL")
    fig.tight_layout(h_pad=0.0)
    # save the figure

    return fig, axis, lifetime, max_abs


def plot_pl_data_with_variance(
    Exp_data, temperature_list, hws, variance_data, save_folder, savefig=False
):
    fig, axes = plt.subplots(1, len(temperature_list) + 1, figsize=(20, 5))
    Exp_data = Exp_data / max(Exp_data.reshape(-1, 1))
    for i in range(len(temperature_list)):
        ax = axes[i]
        ax.plot(
            hws,
            Exp_data[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
            alpha=0.3,
        )
        ax.fill_between(
            hws,
            Exp_data[:, i] - np.sqrt(variance_data[:, i]),
            Exp_data[:, i] + np.sqrt(variance_data[:, i]),
            alpha=0.3,
            color="C" + str(i),
        )
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL Intensity (arb. units)")
        ax.set_title("temperature=" + str(temperature_list[i]) + " K")
        # ax.set_yscale('log')
        ax.set_ylim([0, 1])
    fig.tight_layout()
    if savefig:
        fig.savefig(save_folder + "/data_with_variance.png")
    return fig, axes


if __name__ == "__main__":
    databse_path = Path(
        "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/fit_experimental_emcee_pl/fit_data_base/allLifetimes/"
    )
    test_id = "3410fe8d-1104-4cbf-8421-79be9b38f262"
    main(databse_path, test_id)
