
""" code to estimate the paramter of the model that reproduce the PL data using EMCEE"""
# define a pytensor Op for our likelihood function
from pl_temp_fit import Exp_data_utils

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import os
import emcee
import numpy as np
from pl_temp_fit import LTPL
from scipy.optimize import minimize
from multiprocessing import Pool
import time



def main(
    sigma=0.001,
    data_file="data.csv",
    Temp_std_err=2,
    hws_std_err=0.002,
    relative_intensity_std_error=0.1,
):
    # import data and get model parameters
    number_free_parameters = 5
    Exp_data, temperature_list, hws = Exp_data_utils.read_data(data_file)
    model_config = {
        "number_free_parameters": number_free_parameters,
        "sigma": sigma,
        "Temp_std_err": Temp_std_err,
        "hws_std_err": hws_std_err,
        "relative_intensity_std_error": relative_intensity_std_error,
    }
    X = {'temperature_list':temperature_list, 'hws':hws}
    print(f"size of hw is {hws.shape}")
    print(f"size of temperature_list is {temperature_list.shape}")
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    # generate the data
    save_folder = (
        f"fit_experimental_emcee/{date}/{data_file.split('/')[-1]}/"
        + " sigma=" + str(sigma)
        + " temperature_list=" + str(len(temperature_list))
        + " number_free_parameters=" + str(number_free_parameters)
        + " Temp_std_err="+str(Temp_std_err)
        + " hws_std_err="+str(hws_std_err)
        + " relative_intensity_std_error="+str(relative_intensity_std_error)
    )
    os.makedirs(save_folder, exist_ok=True)
    # get initial covariance matrix
    #get covariance matrix for the experimental data
    init_params = [hws[np.argmax(Exp_data[:,0])], 0.02, 0.1, 0.1, 0.16]
    co_var_mat = plot_generated_data( temperature_list, hws, save_folder, model_config, savefig=True,true_parameters=init_params)
    #plot data with variance
    variance_data = co_var_mat.diagonal().reshape(hws.shape[0],-1).copy()
    Exp_data_utils.plot_PL_data_with_variance(Exp_data, temperature_list, hws, variance_data, save_folder)
    # get maximum likelihood estimate
    soln = get_maximum_likelihood_estimate(Exp_data, co_var_mat,X,save_folder,init_params=init_params)
    # add noise to the data and plot it now with the fitted parameters
    co_var_mat = plot_generated_data( temperature_list, hws, save_folder, model_config, savefig=True, true_parameters=soln.x)
    variance_data = co_var_mat.diagonal().reshape(hws.shape[0],-1).copy()
    Exp_data_utils.plot_PL_data_with_variance(Exp_data, temperature_list, hws, variance_data, save_folder)
    ## run the sampling
    sampler = run_sampler_parallel(save_folder, soln, Exp_data, co_var_mat, X)
    # plot the model data from the mean of the posterior   )
    return  sampler


def get_maximum_likelihood_estimate( Exp_data, co_var_mat,X,save_folder,init_params):
    nll = lambda *args: -pl_loglike(*args)
    soln = minimize(nll, init_params, args=(Exp_data, co_var_mat,  X['temperature_list'], X['hws'])
                    ,bounds=((1,2),(0.001,0.03),(0.03,0.2),(0.03,0.2),(0.1,0.2)),tol=1e-3)
    print(soln.x)
    print("Maximum likelihood estimates:")
    print(f"  E = {soln.x[0]:.3f}")
    print(f"  sigma = {soln.x[1]:.3f}")
    print(f"  LI = {soln.x[2]:.3f}")
    print(f"  L0 = {soln.x[3]:.3f}")
    print(f"  H0 = {soln.x[4]:.3f}")
    print("Maximum log likelihood:", soln.fun)
    # print those into a file
    with open(save_folder+"/maximum_likelihood_estimate.txt", "w") as f:
        f.write("Maximum likelihood estimates:\n")
        f.write(f"  E = {soln.x[0]:.3f}\n")
        f.write(f"  sigma = {soln.x[1]:.3f}\n")
        f.write(f"  LI = {soln.x[2]:.3f}\n")
        f.write(f"  L0 = {soln.x[3]:.3f}\n")
        f.write(f"  H0 = {soln.x[4]:.3f}\n")
        f.write(f"Maximum log likelihood: {soln.fun}\n")
    return soln

#get the variance of the data and plot it
def plot_generated_data(temperature_list, hws, save_folder, model_config, savefig=True,true_parameters=None):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    true_model_pl_list = []
    numnber_of_samples = 100
    for x in range(numnber_of_samples):
        truemodel_pl, true_parameters = generate_data(temperature_list, hws,**model_config, true_parameters=true_parameters)
        data_true_plot = truemodel_pl.reshape(len(hws), -1)
        data_true_plot = data_true_plot/max(data_true_plot.reshape(-1, 1))
        true_model_pl_list.append(data_true_plot.reshape(len(hws), len(temperature_list))+np.random.normal(0, 0.01, size=(len(hws), len(temperature_list))))

        for i in range(len(temperature_list)):
            ax[0].plot(
                hws,
                data_true_plot[:, i],
                label="true" + str(temperature_list[i]) + " K",
                linestyle="--",
                color="C" + str(i),
                alpha=0.3,
            )
        ax[0].set_xlabel("Photon Energy (eV)")
        ax[0].set_ylabel("PL Intensity (arb. units)")
        ax[0].set_title("Posterior mean prediction")
    variance = np.var(np.array(true_model_pl_list), axis=0)
    mean_value_plot = np.mean(np.array(true_model_pl_list), axis=0)
    co_var_mat_test = np.cov(np.array(true_model_pl_list).reshape(numnber_of_samples,-1), rowvar=False)
    np.fill_diagonal(co_var_mat_test, co_var_mat_test.diagonal()+model_config['sigma'])
    #plot the generated data
    for i in range(len(temperature_list)):
        
        ax[1].plot(
            hws,
            mean_value_plot[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
        )
        ax[1].fill_between(
            hws,
            np.mean(np.array(true_model_pl_list), axis=0)[:, i]
            - np.sqrt(variance[:, i]),
            np.mean(np.array(true_model_pl_list), axis=0)[:, i]
            + np.sqrt(variance[:, i]),
            alpha=0.3,
            color="C" + str(i),
        )
        ax[1].set_xlabel("Photon Energy (eV)")
        ax[1].set_ylabel("PL Intensity (arb. units)")
        ax[1].set_title("mean prediction with 1 std dev")

    if savefig:
        fig.savefig(save_folder + "/generated_data.png")
    return co_var_mat_test


def generate_data(temperature_list, hws,sigma,Temp_std_err,hws_std_err,relative_intensity_std_error,number_free_parameters,true_parameters=None,**kwargs):
    if true_parameters is None:
        E_true = 1.5
        sigma_true = 0.02
        LI_true = 0.09
        L0_true = 0.1
        H0_true = 0.15
        true_parameters = [E_true, sigma_true, LI_true, L0_true, H0_true]
    else:
        E_true, sigma_true, LI_true, L0_true, H0_true = true_parameters
    # error in the temperature of the sample
    temperature_list = temperature_list+np.random.normal(0, Temp_std_err, len(temperature_list))
    # error in the detection wavelength
    hws = hws+np.random.normal(0, hws_std_err, len(hws))
    truemodel_pl = pl_trial(
        [E_true, sigma_true, LI_true, L0_true, H0_true], temperature_list, hws
    ) 
    # relative intensity error
    relative_intensity_model = np.max(truemodel_pl, axis=0)/max(truemodel_pl.reshape(-1, 1))
    relative_intensity_model_error = relative_intensity_model + np.random.normal(0, relative_intensity_std_error, len(relative_intensity_model))
    relative_intensity_model_error= relative_intensity_model_error/np.max(relative_intensity_model_error)
    truemodel_pl=truemodel_pl * relative_intensity_model_error / relative_intensity_model
    relative_intensity_model = np.max(truemodel_pl, axis=0)/max(truemodel_pl.reshape(-1, 1))
    # uniform error accross the spectrum
    truemodel_pl = truemodel_pl+np.random.normal(0, sigma, size=(len(hws), len(temperature_list)))

    return truemodel_pl, true_parameters


def pl_trial(theta, temperature_list, hws):

    E, sigma, LI, L0, H0 = theta
    data = LTPL.Data()
    data.CT = LTPL.State(
        E=1.1,
        vmhigh=2,
        vmlow=2,
        sigma=0.01,
        numbrstates=20,
        off=1,
        LI=0.1,
        L0=0.1,
        H0=0.15,
        fosc=5,
    )
    data.D.hw = np.arange(0.5, 3, 0.02)
    data.D.T = temperature_list  # np.array([300.0, 150.0, 80.0])
    data.EX = LTPL.State(
        E=E,
        vmhigh=2,
        vmlow=15,
        sigma=sigma,
        numbrstates=20,
        off=0,
        LI=LI,
        L0=L0,
        H0=H0,
        fosc=5,
    )
    LTPL.LTPLCalc(data)
    PL_results = data.D.kr_hw#.reshape(-1, 1)
    PL_results_interp = np.zeros((len(hws), len(temperature_list)))
    for i in range(len(temperature_list)):
        PL_results_interp[:, i] = np.interp(hws, data.D.hw, PL_results[:, i])

    return PL_results_interp# / max(PL_results)


def log_prior(theta):
    E, sigma_E, LI, L0, H0 = theta
    if 1 < E < 2 and 0.001 < sigma_E < 0.03 and 0.03 < LI < 0.2:
        if 0.03 < L0 < 0.2 and 0.1 < H0 < 0.2:
            return 0.0
    return -np.inf


def pl_loglike( theta, data, co_var_mat, temperature_list, hws):
    data = data/np.max(data.reshape(-1, 1))
    model_data = pl_trial(list(theta), temperature_list, hws)
    model_data = model_data/np.max(model_data.reshape(-1, 1))
    data = data.reshape(-1, 1)
    model_data = model_data.reshape(-1, 1)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data).any() or np.isinf(model_data).any():
        print("NaN in model_data")
        return -np.inf
    diff = data - model_data
    diff[np.abs(diff) < 1e-4] = 0
    loglike = -0.5 * np.dot(diff.T, np.dot(np.linalg.inv(co_var_mat), diff))
    return loglike


def log_probability(theta, data, inv_covar ,X):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    log_like = pl_loglike(theta, data, inv_covar, X['temperature_list'], X['hws'])
    return lp + log_like





class hDFBackend_2(emcee.backends.HDFBackend):
    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            g.attrs["has_blobs"] = False

    def _check(self, state, accepted):
        nwalkers, ndim = self.shape
        if state.coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format(
                    (nwalkers, ndim)
                )
            )
        if state.log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if accepted.shape != (nwalkers,):
            raise ValueError(
                "invalid acceptance size; expected {0}".format(nwalkers)
            )
    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)
        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            g["chain"][iteration, :, :] = state.coords
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None and state.blobs.size > 0:
                g["blobs"][iteration, :] = state.blobs
 
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1
    def has_blobs(self):
        return False

# run the sampler

def run_sampler_parallel(save_folder, soln, Exp_data, co_var_mat, X):
    coords = soln.x + [1e-1,1e-3,1e-2,1e-2,1e-2] * np.random.randn(32, 5)
    nwalkers, ndim = coords.shape
    nsteps = 10000
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    backend = hDFBackend_2(filename, name="multi_core")
    backend.reset(nwalkers, ndim)
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Exp_data, co_var_mat, X),pool=pool,backend=backend) 
        start = time.time()
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(coords, iterations=nsteps,blobs0=[]):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
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
        #sampler.sample(pos, iterations = nsteps, progress=True,store=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    return sampler


def run_sampler_single(save_folder, soln, Exp_data, co_var_mat, X,nsteps=10000):
    coords = soln.x + [1e-1,1e-3,1e-2,1e-2,1e-2] * np.random.randn(32, 5)
    nwalkers, ndim = coords.shape
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    backend = hDFBackend_2(filename, name="single_core")
    backend.reset(nwalkers, ndim)
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Exp_data, co_var_mat, X),backend=backend) 
    start = time.time()
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=nsteps, progress=True,blobs0=[]):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
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
    #sampler.sample(pos, iterations = nsteps, progress=True,store=True)
    end = time.time()
    multi_time = end - start
    print("single process took {0:.1f} seconds".format(multi_time))
    #print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    return sampler


# generate a function to go through the parameter of the main function
# and generate a list of parameters to be passed to the main function
def generate_parameter_list(sigma, 
    Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
):
    from itertools import product

    parameter_list = []
    parameter_list.append({
                "sigma": 0.001,
                "Temp_std_err": 5,
                "hws_std_err": 0.001,
                "relative_intensity_std_error": 0.01,
            })
    for k, l, m, n in product(
        sigma,
        Temp_std_err_list,
        hws_std_err_list,
        relative_intensity_std_error_list,
    ):
        parameter_list.append(
            {
                "sigma": k,
                "Temp_std_err": l,
                "hws_std_err": m,
                "relative_intensity_std_error": n,
            }
        )

    return parameter_list

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()

    argparser.add_argument(
        "--test_number", type=int, default=0, help="test number to run"
    )
    argparser.add_argument(
        "-d",
        "--data_file",
        type=str,
        default="data.csv",
        help="the data file",
    )
    test_number = argparser.parse_args().test_number
    print(f"Running test number {test_number}")
    sigma = [0.01,0.002]
    Temp_std_err_list = [2,10]
    hws_std_err_list = [0.005]
    relative_intensity_std_error_list = [0.1,0.01]

    parameter_list = generate_parameter_list(sigma,
        Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
    )

    # print the parameter list
    print(parameter_list[test_number])
    print(len(parameter_list))
    #parameter_list[test_number]["test_number"] = test_number
    main(data_file=argparser.parse_args().data_file,**parameter_list[test_number])  # run the example