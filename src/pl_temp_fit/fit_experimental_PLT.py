""" code to estimate the paramter of the model that reproduce the PL data using PYMC"""
from pl_temp_fit import LTPL
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import os
import matplotlib.pyplot as plt



def main(
    num_samples=50,
    num_tune=10,
    sigma=0.1,
    number_free_parameters=5,
    data_file="data.csv",
    factor_relative_intensity_loss=10,
):
    truemodel_pl, temperature_list, hws = read_data(data_file)
    print (f"size of hw is {hws.shape}")
    print (f"size of temperature_list is {temperature_list.shape}")
    save_folder = (
        "exp_results_2/"+data_file.split('\\')[-1].split('/')[-1].split(".")[0]
        + str(num_samples)
        + " num_tune="
        + str(num_tune)
        + " sigma="
        + str(sigma)
        + " factor_relative_intensity_loss="
        + str(factor_relative_intensity_loss)
    )
    os.makedirs(save_folder, exist_ok=True)
    
    logl = LogLike(pl_loglike, truemodel_pl, sigma, temperature_list, hws, factor_relative_intensity_loss)
    with pm.Model() as model:
        theta = define_model_prior(number_free_parameters)
        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))
        print("calling the sampler")
        # Use custom number of draws to replace the HMC based defaults
        idata_test = pm.sample(
            5,
            tune=2,
            step=pm.Metropolis(),
            return_inferencedata=True,
        )
    plot_posterior_prediction(
        idata_test,
        truemodel_pl,
        temperature_list,
        hws,
        save_folder,
        savefig=True,
    )
    with model:
        idata_mh = pm.sample(
            num_samples,
            tune=num_tune,
            step=pm.Metropolis(),
            return_inferencedata=True,
        )

    plot_trace(idata_mh, save_folder, savefig=True)
    # save the data
    idata_mh.to_netcdf(save_folder + "/idata.nc")
    # plot the model data from the mean of the posterior
    plot_posterior_prediction(
        idata_mh,
        truemodel_pl,
        temperature_list,
        hws,
        save_folder,
        savefig=True,
    )
    return idata_mh, model


def read_data(csv_file):
    # read csv data where the first row is the photon energy
    # and the columns are the temperature
    # the first column is temperature

    # read the data
    data = np.genfromtxt(csv_file, delimiter=",")
    # the first row is the photon energy
    hws = np.array(data[0, 1:])
    # the first column is temperature
    temperature_list = np.array(data[1:, 0])
    # the rest of the data is the PL intensity
    truemodel_pl = np.array(data[1:, 1:]).transpose()#.reshape(-1, 1)
    return truemodel_pl, temperature_list, hws


def pl_trial(theta, temperature_list, hws):
    if len(theta) == 2:
        E, LI = theta
        sigma, L0, H0 = 0.02, 0.1, 0.15
    elif len(theta) == 5:
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
    data.D.hw = hws  # np.arange(0.8, 2, 0.02)
    data.D.T = temperature_list  # np.array([300.0, 150.0, 80.0])
    data.EX = LTPL.State(
        E=E,
        vmhigh=5,
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
    PL_results = data.D.kr_hw #.reshape(-1, 1)

    return PL_results # / max(PL_results)


def pl_loglike(theta, data, sigma, temperature_list, hws, factor_relative_intensity_loss=10):
    if type(theta) == np.ndarray:
        theta = theta.tolist()
    # theta[0] = theta[0] * 10.0
    # theta[1] = theta[1] * 0.1
    model_data = pl_trial(list(theta), temperature_list, hws)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data).any() or np.isinf(model_data).any():
        return -np.inf
    # calculate the loss function
    # 
    #for   
    loss = 0
    for i in range(len(temperature_list)):
        norm_data = data[:, i] / max(data[:, i])
        norm_model_data = model_data[:, i] / max(model_data[:, i])
        loss = loss + -(0.5 / sigma**2) *  np.sum((norm_data- norm_model_data) ** 2)  
        relative_intensity = max(data[:, i]) / max(data.reshape(-1, 1))
        relative_intensity_model = max(model_data[:, i]) / max(model_data.reshape(-1, 1))
        loss = loss + -(0.5 / sigma**2) * factor_relative_intensity_loss *  np.sum((relative_intensity - relative_intensity_model) ** 2)

    return loss


def generate_data(temperature_list, hws):
    E_true = 1.5
    sigma_E_true = 0.02
    LI_true = 0.11
    L0_true = 0.1
    H0_true = 0.15
    truemodel_pl = pl_trial(
        [E_true, sigma_E_true, LI_true, L0_true, H0_true], temperature_list, hws
    )
    true_parameters = [E_true, sigma_E_true, LI_true, L0_true, H0_true]
    return truemodel_pl, true_parameters


def define_model_prior(number_free_parameters):
    if number_free_parameters == 2:
        E = pm.Uniform("E", 1, 1.8)
        LI = pm.Normal("LI", 0.12, 0.02)
        theta = pt.as_tensor_variable([E, LI])
    elif number_free_parameters == 5:
        E = pm.Uniform("E", 1.0, 2.0)
        sigma_E = pm.Uniform("sigma_E", 0.005, 0.03)
        LI = pm.TruncatedNormal('LI', mu=0.12, sigma=0.02, lower=0.05,upper=0.2)#Normal("LI", 0.12, 0.01)
        L0 = pm.TruncatedNormal('L0', mu=0.12, sigma=0.02, lower=0.05,upper=0.2)#pm.Normal("L0", 0.12, 0.01)
        H0 = pm.TruncatedNormal('H0', mu=0.16, sigma=0.01, lower=0.1,upper=0.2)#pm.Normal("H0", 0.16, 0.01)
        theta = pt.as_tensor_variable([E, sigma_E, LI, L0, H0])
    return theta


def plot_trace(idata_mh, save_folder, savefig=True):
    axes = az.plot_trace(
        idata_mh,
    )
    fig = axes.ravel()[0].figure
    fig.suptitle("Trace plot " + save_folder.split("/")[-1])
    if savefig:
        fig.savefig(save_folder + "/trace.png")


def plot_posterior_prediction(
    idata_mh, truemodel_pl, temperature_list, hws, save_folder="", savefig=True, title="Posterior prediction"
):
    theta_mean = idata_mh.posterior.mean(dim=["chain", "draw"])

    theta_mean_list = []
    for x in theta_mean.data_vars:
        theta_mean_list.append(
            theta_mean[x].values
        )  # theta_mean = theta_mean.to_dict()["values"]
    data_plot = pl_trial(theta_mean_list, temperature_list, hws)
    data_plot = data_plot.reshape(len(hws), -1)/max(data_plot.reshape(-1, 1))
    data_true_plot = truemodel_pl.reshape(len(hws), -1)/max(truemodel_pl.reshape(-1, 1))
    fig, ax = plt.subplots()
    for i in range(len(temperature_list)):
        ax.plot(
            hws,
            data_plot[:, i],
            label="fit" + str(temperature_list[i]) + " K",
            linestyle="-",
            color="C" + str(i),
            alpha=0.5,
        )
        ax.plot(
            hws,
            data_true_plot[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
        )
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel("PL Intensity (arb. units)")
    ax.set_title(title)
    ax.legend()
    if savefig:
        fig.savefig(save_folder + "/posterior_mean.png")


class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, sigma, temperature_list, hws, factor_relative_intensity_loss):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma
        self.temperature_list = temperature_list
        self.hws = hws
        self.factor_relative_intensity_loss = factor_relative_intensity_loss

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(
            theta, self.data, self.sigma, self.temperature_list, self.hws,
            self.factor_relative_intensity_loss
        )

        outputs[0][0] = np.array(logl)  # output the log-likelihood


# generate a function to go through the parameter of the main function
# and generate a list of parameters to be passed to the main function
def generate_parameter_list(
    num_samples, num_tune, sigma, temperature_list, number_free_parameters
):
    from itertools import product

    parameter_list = []
    for i, j, k, l, m in product(
        num_samples,
        num_tune,
        sigma,
        temperature_list,
        number_free_parameters,
    ):
        parameter_list.append(
            {
                "num_samples": i,
                "num_tune": j,
                "sigma": k,
                "temperature_list": np.array(l),
                "number_free_parameters": m,
            }
        )

    return parameter_list


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument(
        "-t",
        "--num_samples",
        type=int,
        default=1000,
        help="the number of samplesn",
    )
    
    argparser.add_argument(
        "-n",
        "--num_tune",
        type=int,
        default=1000,
        help="the number of tuning steps",
    )

    argparser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=0.02,
        help="the noise standard deviation",
    )
    argparser.add_argument(
        "-d",
        "--data_file",
        type=str,
        default="data.csv",
        help="the data file",
    )
    argparser.add_argument(
        "-f",
        "--factor_relative_intensity_loss",
        type=float,
        default=10,
        help="the factor to multiply the relative intensity loss",
    )
    number_free_parameters =  5
    num_samples = argparser.parse_args().num_samples
    factor_relative_intensity_loss = argparser.parse_args().factor_relative_intensity_loss  
    num_tune = argparser.parse_args().num_tune
    sigma = argparser.parse_args().sigma
    data_file = argparser.parse_args().data_file
    print(f"Running on PyMC v{pm.__version__}")
    print(f"num_samples={num_samples}, num_tune={num_tune}, sigma={sigma}, number_free_parameters={number_free_parameters}, factor_relative_intensity_loss={factor_relative_intensity_loss}")
    print(f"data_file={data_file}")
    idata_mh, model = main(
        num_samples,
        num_tune,
        sigma,
        number_free_parameters,
        data_file,
        factor_relative_intensity_loss
    )

