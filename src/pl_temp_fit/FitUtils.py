from pathlib import Path

import emcee
import numpy as np
import time
from multiprocess import Pool
from pl_temp_fit.data_generators.SpectralDataGeneration import (
    SpectralDataGeneration,
)
import logging
logger = logging.getLogger(__name__)

def get_initial_coords(
    data_generator: SpectralDataGeneration,
    coeff_spread,
    num_coords,
    save_folder,
    restart_sampling,
    name="multi_core",
):
    init_params, min_bound_list, max_bound_list = [], [], []
    counter = 0
    for key in ["EX", "CT", "D"]:
        if data_generator.params_to_fit_init[key] == {}:
            continue
        for key2 in data_generator.params_to_fit_init[key].keys():
            init_params.append(data_generator.params_to_fit_init[key][key2])
            min_bound_list.append(data_generator.min_bounds[key][key2])
            max_bound_list.append(data_generator.max_bounds[key][key2])
            counter += 1
    min_bound_list = np.array(min_bound_list)
    max_bound_list = np.array(max_bound_list)
    num_parameters = counter
    coords = init_params + 0.1 * coeff_spread * (
        max_bound_list - min_bound_list
    ) * np.random.randn(num_coords, num_parameters)
    filename = save_folder + "/sampler.h5"
    backend = emcee.backends.HDFBackend(filename, name=name)
    nwalkers, ndim = coords.shape
    if restart_sampling or not Path(filename).is_file():
        backend.reset(nwalkers, ndim)
    else:
        reader = emcee.backends.HDFBackend(filename, name=name)
        if reader.iteration > 0:
            coords = reader.get_last_sample().coords
            # add a little noise to the initial position
            coords += 1e-3 * np.random.randn(*coords.shape)
    return coords, backend


def get_number_of_cores():
    import multiprocessing

    num_processes = multiprocessing.cpu_count()
    print(f"num_processes = {num_processes}")
    return num_processes


def run_auto_correlation_check(
    sampler,
    autocorr,
    index,
    old_tau,
):
    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    try:
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

        old_tau = tau
    except Exception as e:
        print(e)
        print("error in the autocorrelation time")
    return autocorr, index, old_tau, converged


def run_sampling_in_parallel(
    log_probability_glob,
    coords,
    backend,
    dtype,
    num_processes,
    nsteps,
):
    if num_processes is None:
        num_processes = get_number_of_cores()
    index = 0
    autocorr = np.empty(nsteps)
    old_tau = np.inf
    start = time.time()
    num_walkers, num_dim = coords.shape
    with Pool(processes=num_processes) as pool:
        sampler = emcee.EnsembleSampler(
            num_walkers,
            num_dim,
            log_probability_glob,
            backend=backend,
            pool=pool,
            blobs_dtype=dtype,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
        )
        # Now we'll sample for up to max_n steps
        for _ in sampler.sample(
            coords, iterations=nsteps, progress=True, blobs0=[]
        ):
            # Only check convergence every 100 steps

            if sampler.iteration % 100:
                continue
            autocorr, index, old_tau, converged = run_auto_correlation_check(
                sampler, autocorr, index, old_tau
            )
            logger.info(f"converged = {converged}")
            max_log_prob = np.max(sampler.get_log_prob(flat=True))
            logger.info(f"max_log_prob = {max_log_prob}")
            if converged:
                break
    end = time.time()
    multi_time = end - start
    print(f"multi process took {multi_time:.1f} seconds")
    return sampler


def run_single_process_sampling(
    log_probability_glob,
    coords,
    backend,
    dtype,
    nsteps,
):
    num_walkers, num_dim = coords.shape

    sampler = emcee.EnsembleSampler(
        num_walkers,
        num_dim,
        log_probability_glob,
        backend=backend,
        blobs_dtype=dtype,
    )
    start = time.time()
    index = 0
    autocorr = np.empty(nsteps)
    old_tau = np.inf
    # Now we'll sample for up to max_n steps
    for _ in sampler.sample(
        coords, iterations=nsteps, progress=True, blobs0=[]
    ):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        autocorr, index, old_tau = run_auto_correlation_check(
            sampler, autocorr, index, old_tau
        )
    end = time.time()
    multi_time = end - start
    print(f"single process took {multi_time:.1f} seconds")
