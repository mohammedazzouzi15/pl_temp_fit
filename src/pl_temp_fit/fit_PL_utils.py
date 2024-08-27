from pl_temp_fit.Emcee_utils import ensemble_sampler, hDFBackend_2
import time
from scipy.optimize import minimize
from pl_temp_fit import generate_data_utils, Exp_data_utils
import numpy as np
from multiprocessing import Pool
import os
from pl_temp_fit import covariance_utils, config_utils


def get_maximum_likelihood_estimate(
    Exp_data_PL,
    co_var_mat_PL,
    model_config,
    save_folder,
    coeff_spread=0.1,
    num_coords=5,
    fixed_parameters_dict={},
    params_to_fit={},
    min_bound={},
    max_bound={},
):
    nll = lambda *args: -generate_data_utils.pl_loglike(*args)[0]
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
    for i, coord in enumerate(coords):
        print(f"step {i}")
        soln = minimize(
            nll,
            coord,
            args=(
                Exp_data_PL,
                co_var_mat_PL,
                model_config["temperature_list_PL"],
                model_config["hws_PL"],
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
    Exp_data_PL,
    co_var_mat_PL,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    restart_sampling=True,
):

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
    nwalkers, ndim = coords.shape
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    backend = hDFBackend_2(filename, name="single_core")

    if restart_sampling:
        backend.reset(nwalkers, ndim)
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines

    sampler = ensemble_sampler(
        nwalkers,
        ndim,
        generate_data_utils.log_probability_PL,
        args=(
            Exp_data_PL,
            co_var_mat_PL,
            model_config,
            fixed_parameters_dict,
            params_to_fit,
            min_bound,
            max_bound,
        ),
        backend=backend,
    )
    start = time.time()
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(
        coords, iterations=nsteps, progress=True, blobs0=[]
    ):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        try:
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            print(tau)
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            end = time.time()
        except Exception as e:
            print(e)
            print("error in the autocorrelation time")
    # sampler.sample(pos, iterations = nsteps, progress=True,store=True)
    end = time.time()
    multi_time = end - start
    print("single process took {0:.1f} seconds".format(multi_time))
    # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    return sampler


def run_sampler_parallel(
    save_folder,
    Exp_data_PL,
    co_var_mat_PL,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
    num_processes=None,
    restart_sampling=True,
):
    import emcee

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
    nwalkers, ndim = coords.shape
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    # backend = hDFBackend_2(filename, name="multi_core")
    backend = emcee.backends.HDFBackend(filename, name="multi_core")
    if restart_sampling or os.path.isfile(filename) == False:
        backend.reset(nwalkers, ndim)
    else:
        reader = emcee.backends.HDFBackend(filename, name="multi_core")
        coords = reader.get_last_sample().coords
        # add a little noise to the initial position
        coords += 1e-3 * np.random.randn(*coords.shape)
    if coords.shape[0] != nwalkers:
        raise ValueError(
            "invalid coordinate dimensions; expected {0}".format(
                (nwalkers, ndim)
            )
        )
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines
    if num_processes is None:
        import multiprocessing

        num_processes = multiprocessing.cpu_count()
        print(f"num_processes = {num_processes}")
    dtype = [
        ("log_likelihood", float),
        ("Chi square", float),
        ("Ex_knr", float),
        ("Ex_kr", float),
    ]
    with Pool(processes=num_processes) as pool:

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            generate_data_utils.log_probability_PL,
            args=(
                Exp_data_PL,
                co_var_mat_PL,
                model_config,
                fixed_parameters_dict,
                params_to_fit,
                min_bound,
                max_bound,
            ),
            backend=backend,
            pool=pool,
            blobs_dtype=dtype,
        )
        start = time.time()
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(
            coords, iterations=nsteps, progress=True, blobs0=[]
        ):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                print(tau)
                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
                end = time.time()
            except Exception as e:
                print(e)
                print("error in the autocorrelation time")
        # sampler.sample(pos, iterations = nsteps, progress=True,store=True)
        end = time.time()
        multi_time = end - start
        print("multi process took {0:.1f} seconds".format(multi_time))
        # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    return sampler


def plot_exp_data_with_variance(
    temperature_list_PL,
    hws_PL,
    variance_PL,
    save_folder,
    fixed_parameters_dict,
    true_parameters,
    Exp_data_PL,
    fig=None,
    axis=None,
):
    model_data_PL, EX_kr, Ex_knr = generate_data_utils.pl_trial(
        temperature_list_PL,
        hws_PL,
        fixed_parameters_dict,
        true_parameters,
    )
    truemodel_pl = model_data_PL / np.max(model_data_PL.reshape(-1, 1))
    if fig is None:
        fig, axis = Exp_data_utils.plot_PL_data_with_variance(
            Exp_data_PL, temperature_list_PL, hws_PL, variance_PL, save_folder
        )

    for i, axes in enumerate(axis):
        axes.plot(hws_PL, truemodel_pl[:, i], label="fit", color="C" + str(i))
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
    csv_name = model_config_save['csv_name_PL']
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
    save_folder = model_config_save["save_folder"]
    co_var_mat_PL, variance_PL = covariance_utils.plot_generated_data_PL(
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
            model_config["temperature_list_PL"],
            model_config["hws_PL"],
            variance_PL,
            save_folder,
            fixed_parameters_dict,
            paramers,
            Exp_data,
        )
        fig.suptitle(title_list[_id])
