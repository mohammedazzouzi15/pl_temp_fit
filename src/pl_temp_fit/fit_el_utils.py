import time
from pathlib import Path

import emcee
import numpy as np
from multiprocess import Pool
from scipy.optimize import minimize

from pl_temp_fit import (
    Exp_data_utils,
    config_utils,
    covariance_utils,
    generate_data_utils,
    FitUtils,
)
from pl_temp_fit.Emcee_utils import ensemble_sampler


def get_maximum_likelihood_estimate(
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
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
        nll = -generate_data_utils.el_loglike(*args)[0]
        # check type of nll
        if isinstance(nll, np.ndarray):
            return nll[0][0]
        if isinstance(nll, list):
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
    inv_cov_el = np.linalg.inv(co_var_mat_el)
    inv_cov_pl = np.linalg.inv(co_var_mat_pl)
    for i, coord in enumerate(coords):
        print(f"step {i}")
        soln = minimize(
            nll,
            coord,
            args=(
                Exp_data_el,
                Exp_data_pl,
                inv_cov_el,
                inv_cov_pl,
                model_config["temperature_list_el"],
                model_config["hws_el"],
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
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    restart_sampling=False,
):
    coords, backend = FitUtils.get_initial_coords(
        params_to_fit,
        min_bound,
        max_bound,
        coeff_spread,
        num_coords,
        save_folder,
        restart_sampling,
        name="single_core",
    )
    nwalkers, ndim = coords.shape
    # Here are the important lines
    inv_cov_el = np.linalg.inv(co_var_mat_el)
    inv_cov_pl = np.linalg.inv(co_var_mat_pl)

    def log_probability_glob(theta):
        return generate_data_utils.log_probability(
            theta,
            Exp_data_el,
            Exp_data_pl,
            inv_cov_el,
            inv_cov_pl,
            model_config,
            fixed_parameters_dict,
            params_to_fit,
            min_bound,
            max_bound,
        )

    return FitUtils.run_single_process_sampling(
        log_probability_glob,
        coords,
        backend,
        nsteps,
        dtype=[
            ("log_likelihood", float),
            ("Chi square", float),
            ("DVNR", float),
            ("CT_knr", float),
            ("CT_kr", float),
            ("EX_knr", float),
            ("EX_kr", float),
        ],
    )


def run_sampler_parallel(
    save_folder,
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    num_processes=None,
    restart_sampling=False,
):
    coords, backend = FitUtils.get_initial_coords(
        params_to_fit,
        min_bound,
        max_bound,
        coeff_spread,
        num_coords,
        save_folder,
        restart_sampling,
    )
    nwalkers, ndim = coords.shape

    inv_cov_el = np.linalg.inv(co_var_mat_el)
    inv_cov_pl = np.linalg.inv(co_var_mat_pl)
    # Here are the important lines

    if num_processes is None:
        num_processes = FitUtils.get_number_of_cores()

    def log_probability_glob(theta):
        return generate_data_utils.log_probability(
            theta,
            Exp_data_el,
            Exp_data_pl,
            inv_cov_el,
            inv_cov_pl,
            model_config,
            fixed_parameters_dict,
            params_to_fit,
            min_bound,
            max_bound,
        )

    dtype = [
        ("log_likelihood", float),
        ("Chi square", float),
        ("DVNR", float),
        ("CT_knr", float),
        ("CT_kr", float),
        ("EX_knr", float),
        ("EX_kr", float),
    ]

    return FitUtils.run_sampling_in_parallel(
        log_probability_glob,
        coords,
        backend,
        dtype,
        num_processes,
        nsteps,
    )


def plot_exp_data_with_variance(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    variance_el,
    variance_pl,
    save_folder,
    fixed_parameters_dict,
    true_parameters,
    Exp_data_pl,
    Exp_data_el,
):
    model_data_el, model_data_pl, dv_nr, CT_knr, CT_kr, EX_knr, EX_kr = generate_data_utils.el_trial(
        temperature_list_el,
        hws_el,
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        true_parameters,
    )
    truemodel_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    truemodel_el = model_data_el / np.max(model_data_el.reshape(-1, 1))
    fig, axis = Exp_data_utils.plot_pl_data_with_variance(
        Exp_data_el, temperature_list_el, hws_el, variance_el, save_folder
    )

    for i, axes in enumerate(axis):
        axes.plot(hws_el, truemodel_el[:, i], label="fit", color="C" + str(i))
        axes.legend()
        axes.set_ylim(0, 1.1)
        axes.set_ylabel("EL intensity")
    fig.suptitle("EL")
    fig.tight_layout(h_pad=0.0)
    fig, axis = Exp_data_utils.plot_pl_data_with_variance(
        Exp_data_pl, temperature_list_pl, hws_pl, variance_pl, save_folder
    )

    for i, axes in enumerate(axis):
        axes.plot(hws_pl, truemodel_pl[:, i], label="fit", color="C" + str(i))
        axes.legend()
        axes.set_ylim(0, 1.1)
    fig.suptitle("PL")
    fig.tight_layout(h_pad=0.0)
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
    exp_data_pl, temperature_list, hws = Exp_data_utils.read_data(
        model_config_save["csv_name_pl"]
    )
    exp_data_el, temperature_list, hws = Exp_data_utils.read_data(
        model_config_save["csv_name_el"]
    )
    save_folder = model_config_save["save_folder"]
    co_var_mat_pl, co_var_mat_el, variance_el, variance_pl = (
        covariance_utils.plot_generated_data(
            save_folder,
            model_config,
            savefig=True,
            fixed_parameters_dict=fixed_parameters_dict,
            params_to_fit=params_to_fit,
        )
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
            model_config["temperature_list_el"],
            model_config["hws_el"],
            variance_el,
            variance_pl,
            save_folder,
            fixed_parameters_dict,
            paramers,
            exp_data_pl,
            exp_data_el,
        )
        fig.suptitle(title_list[_id])
