import numpy as np
from scipy.optimize import minimize

from pl_temp_fit import (
    Exp_data_utils,
    FitUtils,
    config_utils,
    covariance_utils,
    generate_data_utils,
)


def get_maximum_likelihood_estimate(
    Exp_data_pl,
    co_var_mat_pl,
    model_config,
    save_folder,
    coeff_spread=0.1,
    num_coords=5,
    fixed_parameters_dict={},
    params_to_fit={},
    min_bound={},
    max_bound={},
):
    def nll(*args):
        nll = -generate_data_utils.pl_loglike(*args)[0]
        # check type of nll
        if isinstance(nll, np.ndarray):
            return nll[0][0]
        return nll

    init_params, min_bound_list, max_bound_list = [], [], []
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            init_params.append(params_to_fit[key][key2])
            min_bound_list.append(min_bound[key][key2])
            max_bound_list.append(max_bound[key][key2])
            counter += 1
    min_bound_list = np.array(min_bound_list)
    max_bound_list = np.array(max_bound_list)
    num_parameters = counter
    coords = init_params + 0.1 * coeff_spread * (
        max_bound_list - min_bound_list
    ) * np.random.randn(num_coords, num_parameters)
    min_fun = np.inf
    print("running the minimisation")
    inv_covar_mat = np.linalg.inv(co_var_mat_pl)

    for i, coord in enumerate(coords):
        print(f"step {i}")
        soln = minimize(
            nll,
            coord,
            args=(
                Exp_data_pl,
                inv_covar_mat,
                model_config["temperature_list_pl"],
                model_config["hws_pl"],
                fixed_parameters_dict,
                params_to_fit,
            ),
            bounds=[
                (min_bound_list[i], max_bound_list[i])
                for i in range(num_parameters)
            ],
            tol=1e-2,
        )
        if "NORM_OF_PROJECTED_GRADIENT_" in soln.message:
            print("NORM_OF_PROJECTED_GRADIENT_")
            print(soln.x)
            print(soln.fun)
            continue
        if soln.fun < min_fun:
            min_fun = soln.fun
            soln_min = soln
    print(soln_min.x)
    print("Maximum likelihood estimates:")
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            print(f"  {key}_{key2} = {soln_min.x[counter]:.3f}")
            counter += 1
    print("Maximum log likelihood:", soln.fun)
    # print those into a file
    with open(save_folder + "/maximum_likelihood_estimate.txt", "w") as f:
        f.write("Maximum likelihood estimates:\n")
        counter = 0
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue
            for key2 in params_to_fit[key].keys():
                f.write(f"  {key}_{key2} = {soln_min.x[counter]:.3f}")
                counter += 1

        f.write(f"Maximum log likelihood: {soln_min.fun}\n")
    return soln_min


def run_sampler_single(
    save_folder,
    Exp_data_pl,
    co_var_mat_pl,
    data_generator,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    restart_sampling=True,
):
    coords, backend = FitUtils.get_initial_coords(
        data_generator,
        coeff_spread,
        num_coords,
        save_folder,
        restart_sampling,
        name="single_core",
    )

    inv_cov_pl = np.linalg.inv(co_var_mat_pl)
    dtype = data_generator.dtypes

    def log_probability_glob(theta):
        return data_generator.log_probability(
            theta,
            Exp_data_pl,
            inv_cov_pl,
        )

    return FitUtils.run_single_process_sampling(
        log_probability_glob,
        coords,
        backend,
        dtype=dtype,
        nsteps=nsteps,
    )


def run_sampler_parallel(
    save_folder,
    Exp_data_pl,
    co_var_mat_pl,
    data_generator,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    num_processes=None,
    restart_sampling=True,
):
    coords, backend = FitUtils.get_initial_coords(
        data_generator,
        coeff_spread,
        num_coords,
        save_folder,
        restart_sampling,
    )

    inv_cov_pl = np.linalg.inv(co_var_mat_pl)
    # Here are the important lines
    if num_processes is None:
        num_processes = FitUtils.get_number_of_cores()
    dtype = data_generator.dtypes

    def log_probability_glob(theta):
        return data_generator.log_probability(
            theta,
            Exp_data_pl,
            inv_cov_pl,
        )

    return FitUtils.run_sampling_in_parallel(
        log_probability_glob,
        coords,
        backend,
        dtype,
        num_processes,
        nsteps,
    )


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
        return pl_results_interp, abs_results_interp

    model_data_pl, abs_results_interp = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        true_parameters,
    )
    truemodel_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    if fig is None:
        fig, axis = Exp_data_utils.plot_pl_data_with_variance(
            Exp_data_pl, temperature_list_pl, hws_pl, variance_pl, save_folder
        )

    for i, axes in enumerate(axis):
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

    return fig, axis


def get_param_dict(params_to_fit_init, true_params_list):
    true_parameters = {
        "EX": {},
        "CT": {},
        "D": {},
    }
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit_init[key] == {}:
            continue
        for id, key2 in enumerate(params_to_fit_init[key].keys()):
            true_parameters[key][key2] = true_params_list[counter]
            counter += 1
    return true_parameters


def plot_fit_limits(model_config, model_config_save):
    fixed_parameters_dict, params_to_fit, min_bound, max_bound = (
        config_utils.get_dict_params(model_config_save)
    )
    csv_name = model_config_save["csv_name_pl"]
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    save_folder = model_config_save["save_folder"]
    co_var_mat_pl, variance_pl = covariance_utils.plot_generated_data_pl(
        save_folder,
        model_config,
        savefig=True,
        fixed_parameters_dict=fixed_parameters_dict,
        params_to_fit=model_config_save["params_to_fit_init"],
    )
    title_list = ["Initial Parameters", "Min Bound", "Max Bound"]
    for _id, paramers in enumerate(
        [
            model_config_save["params_to_fit_init"],
            model_config_save["min_bounds"],
            model_config_save["max_bounds"],
        ]
    ):
        fig, axis = plot_exp_data_with_variance(
            model_config["temperature_list_pl"],
            model_config["hws_pl"],
            variance_pl,
            save_folder,
            fixed_parameters_dict,
            paramers,
            Exp_data,
        )
        fig.suptitle(title_list[_id])
        fig.savefig(save_folder + f"/PL_fit{_id}.png")


def plot_fit_limits_overlay(model_config, model_config_save):
    """Overlay the results for initial, min, and max parameters on the same plot for each temperature."""
    fixed_parameters_dict, params_to_fit, min_bound, max_bound = (
        config_utils.get_dict_params(model_config_save)
    )
    csv_name = model_config_save["csv_name_pl"]
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    save_folder = model_config_save["save_folder"]
    co_var_mat_pl, variance_pl = covariance_utils.plot_generated_data_pl(
        save_folder,
        model_config,
        savefig=False,
        fixed_parameters_dict=fixed_parameters_dict,
        params_to_fit=model_config_save["params_to_fit_init"],
    )
    param_sets = [
        (model_config_save["params_to_fit_init"], "Initial", "C0", "-"),
        (model_config_save["min_bounds"], "Min Bound", "C1", "--"),
        (model_config_save["max_bounds"], "Max Bound", "C2", ":"),
    ]
    # Get model results for each parameter set
    pl_results_list = []
    abs_results_list = []
    for paramers, label, color, style in param_sets:
        # Use the pl_trial logic from plot_exp_data_with_variance
        from pl_temp_fit.model_function import LTL

        def pl_trial(
            temperature_list_pl,
            hws_pl,
            fixed_parameters_dict={},
            params_to_fit={},
        ):
            data = LTL.Data()
            data.update(**fixed_parameters_dict)
            data.update(**params_to_fit)
            data.D.Luminecence_exp = "PL"
            data.D.T = temperature_list_pl
            LTL.ltlcalc(data)
            pl_results = data.D.kr_hw
            pl_results_interp = np.zeros(
                (len(hws_pl), len(temperature_list_pl))
            )
            abs_results_interp = np.zeros(
                (len(hws_pl), len(temperature_list_pl))
            )
            for i in range(len(temperature_list_pl)):
                pl_results_interp[:, i] = np.interp(
                    hws_pl, data.D.hw, pl_results[:, i]
                )
                abs_results_interp[:, i] = np.interp(
                    hws_pl, data.D.hw, data.D.alpha[:, i]
                )
            pl_results_interp = (
                pl_results_interp
                / pl_results_interp[pl_results_interp > 0].max()
            )
            abs_results_interp = (
                abs_results_interp / abs_results_interp.reshape(-1).max()
            )
            return pl_results_interp, abs_results_interp

        pl_results, abs_results = pl_trial(
            model_config["temperature_list_pl"],
            model_config["hws_pl"],
            fixed_parameters_dict,
            paramers,
        )
        pl_results_list.append((pl_results, label, color, style))
        abs_results_list.append((abs_results, label, color, style))
    # Plot overlay
    fig, axis = Exp_data_utils.plot_pl_data_with_variance(
        Exp_data,
        model_config["temperature_list_pl"],
        model_config["hws_pl"],
        variance_pl,
        save_folder,
    )
    for i, axes in enumerate(axis):
        for pl_results, label, color, style in pl_results_list:
            axes.plot(
                model_config["hws_pl"],
                pl_results[:, i],
                label=f"PL {label}",
                color=f"C{i}",
                linestyle=style,
            )
        
        axes.legend()
        axes.set_ylim(0, 1.1)
    fig.suptitle("PL Fit Limits Overlay")
    fig.tight_layout(h_pad=0.0)
    fig.savefig(save_folder + "/PL_fit_limits_overlay.png")
    return fig, axis
