
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
from pl_temp_fit import LTL
from scipy.optimize import minimize
from multiprocessing import Pool
import time



def main(
    sigma=0.001,
    data_file_PL="data_PL.csv",
    data_file_EL="data_EL.csv",
    Temp_std_err=2,
    hws_std_err=0.002,
    relative_intensity_std_error_PL=0.1,
    relative_intensity_std_error_EL=0.1,
    LE_params = [1.37,1e-03,7.87e-02,1.1e-01, 1.59e-01],
    coeff_spread = 10
):
    # import data and get model parameters
    number_free_parameters = 5
    Exp_data_PL, temperature_list_PL, hws_PL = Exp_data_utils.read_data(data_file_PL)
    Exp_data_EL, temperature_list_EL, hws_EL = Exp_data_utils.read_data(data_file_EL)
    #initialise parameters for the model
    number_free_parameters , sigma, Temp_std_err, hws_std_err, relative_intensity_std_error_PL,relative_intensity_std_error_EL = 5, 0.001, 10, 0.005, 0.05,0.1

    model_config = {
            "number_free_parameters": number_free_parameters,
            "sigma": sigma,
            "Temp_std_err": Temp_std_err,
            "hws_std_err": hws_std_err,
            "relative_intensity_std_error_PL": relative_intensity_std_error_PL,
            "relative_intensity_std_error_EL": relative_intensity_std_error_EL,
        }
    X = {'temperature_list_PL':temperature_list_PL, 'hws_PL':hws_PL,
        'temperature_list_EL':temperature_list_EL, 'hws_EL':hws_EL}
    print(f"size of hw is {hws_PL.shape}")
    print(f"size of temperature_list is {temperature_list_PL.shape}")
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    # generate the data
    save_folder = (
        f"fit_experimental_emcee_EL/{date}/{data_file_PL.split('/')[-1].split('.')[0]}/"
        + "_sigma=" + str(sigma)
        + "_temperature_list=" + str(len(temperature_list_PL))
        + "_numb_parms=" + str(number_free_parameters)
        + "_Temp_std_err="+str(Temp_std_err)
        + "_hws_std_err="+str(hws_std_err)
        + "_coeff_spread="+str(coeff_spread)
    # + "relative_intensity_std_error_PL="+str(relative_intensity_std_error_PL)
    # + "relative_intensity_std_error_EL="+str(relative_intensity_std_error_EL)
    )
    os.makedirs(save_folder, exist_ok=True)
    # get initial covariance matrix
    #get covariance matrix for the experimental data
    #LE_params = [1.37,1e-03,7.87e-02,1.1e-01, 1.59e-01] # this needs to be set from the beginning
    init_params = [hws_PL[np.argmax(Exp_data_PL[:,0])]-0.1, 10, 0.1, 0.1, 0.16,-2]
    co_var_mat_PL,co_var_mat_EL,variance_EL,variance_PL= plot_generated_data(temperature_list_EL, hws_EL,temperature_list_PL,hws_PL, save_folder, model_config,LE_params, savefig=True,true_parameters=init_params)

    # get maximum likelihood estimate
    soln = get_maximum_likelihood_estimate( Exp_data_EL,Exp_data_PL, co_var_mat_PL,co_var_mat_EL ,X,save_folder,LE_params,init_params)
    # add noise to the data and plot it now with the fitted parameters
    co_var_mat_PL,co_var_mat_EL,variance_EL,variance_PL= plot_generated_data(temperature_list_EL, hws_EL,temperature_list_PL,hws_PL, save_folder, model_config,LE_params, savefig=True,true_parameters=soln.x)

    ## run the sampling
    sampler = run_sampler_parallel(save_folder, soln, Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL,LE_params, X,nsteps=100000,coeff_spread=coeff_spread)
    # plot the model data from the mean of the posterior   )
    return  sampler




#get the variance of the data and plot it
def plot_generated_data(temperature_list_EL, hws_EL,temperature_list_PL,hws_PL, save_folder, model_config,LE_params, savefig=True,true_parameters=None):
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    true_model_pl_list = []
    true_model_el_list = []
    numnber_of_samples = 10
    for x in range(numnber_of_samples):
        model_data_PL,model_data_EL, true_parameters= generate_data(temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,LE_params,**model_config,true_parameters=true_parameters)
        def add_plot_to_ax(model_data,hws, temperature_list,ax,true_model_list):
            data_true_plot = model_data.reshape(len(hws), -1)
            data_true_plot = data_true_plot/max(data_true_plot.reshape(-1, 1))
            true_model_list.append(data_true_plot.reshape(len(hws), len(temperature_list))+np.random.normal(0, 0.01, size=(len(hws), len(temperature_list))))

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
            return true_model_list
        true_model_pl_list = add_plot_to_ax(model_data_PL, hws_PL, temperature_list_PL,ax[0][0],true_model_pl_list)
        true_model_el_list = add_plot_to_ax(model_data_EL, hws_EL, temperature_list_EL,ax[0][1],true_model_el_list)
    def get_covariance_matrix(true_model_pl_list):
        variance = np.var(np.array(true_model_pl_list), axis=0)
        co_var_mat_test = np.cov(np.array(true_model_pl_list).reshape(numnber_of_samples,-1), rowvar=False)
        np.fill_diagonal(co_var_mat_test, co_var_mat_test.diagonal()+model_config['sigma'])
        return co_var_mat_test,variance
    co_var_mat_test_PL,variance_PL = get_covariance_matrix(true_model_pl_list)
    co_var_mat_test_EL,variance_EL = get_covariance_matrix(true_model_el_list)
    def plot_mean_and_variance(ax,co_var_mat_test,variance,hws,temperature_list,true_model_pl_list):
        mean_value_plot = np.mean(np.array(true_model_pl_list), axis=0)
        print(f"shape of mean value plot is {mean_value_plot.shape}")
        #plot the generated data
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
    co_var_mat_test_PL = plot_mean_and_variance(ax[1][0],co_var_mat_test_PL,variance_PL,hws_PL,temperature_list_PL,true_model_pl_list)
    co_var_mat_test_EL = plot_mean_and_variance(ax[1][1],co_var_mat_test_EL,variance_EL,hws_EL,temperature_list_EL,true_model_el_list)
    #plot the generated data
    if savefig:
        fig.savefig(save_folder + "/generated_data.png")
    fig.tight_layout()
    return co_var_mat_test_PL,co_var_mat_test_EL,variance_EL,variance_PL


def generate_data(temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,LE_params,sigma,Temp_std_err,hws_std_err,relative_intensity_std_error_PL,relative_intensity_std_error_EL,number_free_parameters,true_parameters=None):
    if true_parameters is None:

        true_parameters = [1.2,10,0.1,0.1,0.16,-2]
    # error in the temperature of the sample
        
    temperature_list_PL = temperature_list_PL+np.random.normal(0, Temp_std_err, len(temperature_list_PL))
    temperature_list_EL = temperature_list_EL+np.random.normal(0, Temp_std_err, len(temperature_list_EL))

    # error in the detection wavelength
    hws_PL = hws_PL+np.random.normal(0, hws_std_err, len(hws_PL))
    hws_EL = hws_EL+np.random.normal(0, hws_std_err, len(hws_EL))
    model_data_EL, model_data_PL= el_trial(list(true_parameters), temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,LE_params)

    # relative intensity error
    def add_relative_intensity_error(model_data_PL, relative_intensity_std_error):
        relative_intensity_model = np.max(model_data_PL, axis=0)/max(model_data_PL.reshape(-1, 1))
        relative_intensity_model_error = relative_intensity_model + np.random.normal(0, relative_intensity_std_error, len(relative_intensity_model))
        relative_intensity_model_error= np.abs(relative_intensity_model_error/np.max(relative_intensity_model_error))
        model_data_PL=model_data_PL * relative_intensity_model_error / relative_intensity_model
        return model_data_PL
    model_data_PL = add_relative_intensity_error(model_data_PL, relative_intensity_std_error_PL)
    model_data_EL = add_relative_intensity_error(model_data_EL, relative_intensity_std_error_EL)

    return model_data_PL,model_data_EL, true_parameters


def get_maximum_likelihood_estimate( Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL,X,save_folder,LE_params,init_params):
    nll = lambda *args: -el_loglike(*args)
    soln = minimize(nll, init_params, args=( Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL, X['temperature_list_EL'], X['hws_EL'],X['temperature_list_PL'], X['hws_PL'],LE_params)
                    ,bounds=((1,2),(8,12),(0.03,0.2),(0.03,0.2),(0.1,0.2),(-4,-1)),tol=1e-3)
    print(soln.x)
    print("Maximum likelihood estimates:")
    print(f"  E_CT = {soln.x[0]:.3f}")
    print(f"  K_EXCT = {soln.x[1]:.3f}")
    print(f"  LI = {soln.x[2]:.3f}")
    print(f"  L0 = {soln.x[3]:.3f}")
    print(f"  H0 = {soln.x[4]:.3f}")
    print(f"  fosc_ct = {soln.x[5]:.3f}")

    print("Maximum log likelihood:", soln.fun)
    # print those into a file
    with open(save_folder+"/maximum_likelihood_estimate.txt", "w") as f:
        f.write("Maximum likelihood estimates:\n")
        f.write(f"  E_CT = {soln.x[0]:.3f}\n")
        f.write(f"  K_EXCT = {soln.x[1]:.3f}\n")
        f.write(f"  LI = {soln.x[2]:.3f}\n")
        f.write(f"  L0 = {soln.x[3]:.3f}\n")
        f.write(f"  H0 = {soln.x[4]:.3f}\n")
        f.write(f"  fosc_ct = {soln.x[5]:.3f}\n")

        f.write(f"Maximum log likelihood: {soln.fun}\n")
    return soln



def el_trial(theta, temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,theta_Ex):

    E, k_EXCT, LI, L0, H0, fosc_CT= theta
    E_EX, sigma_E, LI_EX, L0_EX, H0_EX= theta_Ex
    data = LTL.Data()
    data.D.RCTE = 1
    data.CT = LTL.State(
        E=E,
        vmhigh=2,
        vmlow=15,
        sigma=0.01,
        numbrstates=20,
        off=0,
        LI=LI,
        L0=L0,
        H0=H0,
        fosc=10**fosc_CT,
    )
    data.D.hw = np.arange(0.5, 5, 0.02)
    data.D.T = temperature_list_EL  # np.array([300.0, 150.0, 80.0])
    data.EX = LTL.State(
        E=E_EX,
        vmhigh=2,
        vmlow=15,
        sigma=sigma_E,
        numbrstates=20,
        off=0,
        LI=LI_EX,
        L0=L0_EX,
        H0=H0_EX,
        fosc=5,
    )
    data.D.kEXCT = 10**k_EXCT #* np.exp(-(data.EX.E - data.CT.E)**2 * ((data.c.kb * 300) / 0.01 / (data.c.kb * data.D.T)))
    data.D.Luminecence_exp = 'EL'
    LTL.LTLCalc(data)
    EL_results = data.D.kr_hw#.reshape(-1, 1)
    EL_results_interp = np.zeros((len(hws_EL), len(temperature_list_EL)))
    for i in range(len(temperature_list_EL)):
        EL_results_interp[:, i] = np.interp(hws_EL, data.D.hw, EL_results[:, i])
    data.D.Luminecence_exp = 'PL'
    data.D.T = temperature_list_PL  # np.array([300.0, 150.0, 80.0])

    LTL.LTLCalc(data)
    PL_results = data.D.kr_hw#.reshape(-1, 1)
    PL_results_interp = np.zeros((len(hws_PL), len(temperature_list_PL)))
    for i in range(len(temperature_list_PL)):
        PL_results_interp[:, i] = np.interp(hws_PL, data.D.hw, PL_results[:, i])
    return EL_results_interp,PL_results_interp# / max(PL_results)




def log_prior(theta,E_ex):
    E_CT, k_EXCT, LI, L0, H0,fosc_CT = theta
    if 1 < E_CT <E_ex-0.03  and 8 < k_EXCT < 12 and 0.03 < LI < 0.2:
        if 0.03 < L0 < 0.2 and 0.1 < H0 < 0.2 and -4 < fosc_CT < -1:
            return 0.0
    return -np.inf


def el_loglike( theta, data_EL,data_PL, co_var_mat_EL,co_var_mat_PL, temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,LE_params):
    model_data_EL, model_data_PL= el_trial(list(theta), temperature_list_EL, hws_EL,temperature_list_PL,hws_PL,LE_params)
    model_data_EL = model_data_EL/np.max(model_data_EL.reshape(-1, 1))
    model_data_EL = model_data_EL.reshape(-1, 1)
    data_EL = data_EL/np.max(data_EL.reshape(-1, 1))
    data_EL = data_EL.reshape(-1, 1)
    model_data_PL = model_data_PL/np.max(model_data_PL.reshape(-1, 1))
    model_data_PL = model_data_PL.reshape(-1, 1)
    data_PL = data_PL/np.max(data_PL.reshape(-1, 1))
    data_PL = data_PL.reshape(-1, 1)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data_EL).any() or np.isinf(model_data_EL).any():
        print("NaN in model_data")
        return -np.inf
    diff_EL = data_EL - model_data_EL
    diff_EL[np.abs(diff_EL) < 1e-4] = 0
    loglike = -0.5 * np.dot(diff_EL.T, np.dot(np.linalg.inv(co_var_mat_EL), diff_EL))
    diff_PL = data_PL - model_data_PL
    diff_PL[np.abs(diff_PL) < 1e-4] = 0
    loglike = loglike -0.5 * np.dot(diff_PL.T, np.dot(np.linalg.inv(co_var_mat_PL), diff_PL))

    return loglike


def log_probability(theta, data_EL,data_PL, co_var_mat_EL,co_var_mat_PL ,X,LE_params):
    lp = log_prior(theta,LE_params[0])
    if lp < -1e9:
        print("log prior is -inf")
        return -np.inf
    log_like = el_loglike(theta, data_EL,data_PL, co_var_mat_EL,co_var_mat_PL, X['temperature_list_EL'], X['hws_EL'],X['temperature_list_PL'], X['hws_PL'],LE_params)
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

def run_sampler_parallel(save_folder, soln, Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL,LE_params, X,nsteps = 10000,coeff_spread = 10):
    coords = soln.x + [coeff_spread*1e-2,coeff_spread*1e-1,coeff_spread*1e-2,coeff_spread*1e-2,coeff_spread*1e-2,coeff_spread*1e-1] * np.random.randn(32, 6)
    nwalkers, ndim = coords.shape
    
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL, X,LE_params),pool=pool,backend=backend) 
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


def run_sampler_single(save_folder, soln, Exp_data_EL, Exp_data_PL,
                        co_var_mat_EL, co_var_mat_PL,LE_params, X,nsteps=10000,    coeff_spread = 10):

    coords = soln.x + [coeff_spread*1e-2,coeff_spread*1e-1,coeff_spread*1e-2,coeff_spread*1e-2,coeff_spread*1e-2,coeff_spread*1e-1] * np.random.randn(32, 6)   
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

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=( Exp_data_EL,Exp_data_PL, co_var_mat_EL,co_var_mat_PL, X,LE_params),backend=backend) 
    start = time.time()
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=nsteps, progress=True,blobs0=[]):
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
                "relative_intensity_std_error_PL": 0.01,
                "relative_intensity_std_error_EL": 0.01,
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
                "relative_intensity_std_error_PL": n,
                "relative_intensity_std_error_EL": n,
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
        "-dPL",
        "--data_file_PL",
        type=str,
        default="data.csv",
        help="the data file",
    )
    argparser.add_argument(
        "-dEL",
        "--data_file_EL",
        type=str,
        default="data.csv",
        help="the data file",
    )
    argparser.add_argument(
        "--coeff_spread",
        type=int,
        default=10,
        help="the spread of the coefficient"
    )
    test_number = argparser.parse_args().test_number
    print(f"Running test number {test_number}")
    sigma = [0.01,0.002]
    Temp_std_err_list = [2,10]
    hws_std_err_list = [0.05,0.005]
    relative_intensity_std_error_list = [0.1,0.01]

    parameter_list = generate_parameter_list(sigma,
        Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
    )

    # print the parameter list
    print(parameter_list[test_number])
    print(len(parameter_list))
    #parameter_list[test_number]["test_number"] = test_number
    main(data_file_PL=argparser.parse_args().data_file_PL,
         data_file_EL=argparser.parse_args().data_file_EL,
         coeff_spread=argparser.parse_args().coeff_spread,
         **parameter_list[test_number])  # run the example