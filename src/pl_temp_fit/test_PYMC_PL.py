# define a pytensor Op for our likelihood function
from pl_temp_fit import LTPL
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import os
import matplotlib.pyplot as plt

print(f"Running on PyMC v{pm.__version__}")


def main(
    num_samples=50,
    num_tune=10,
    sigma=0.1,
    temperature_list=np.array([300.0, 150.0, 80.0]),
    number_free_parameters=2,
    hws=np.arange(0.8, 2, 0.02),
):
    truemodel_pl, true_parameters = generate_data(temperature_list, hws,sigma)
    logl = LogLike(pl_loglike, truemodel_pl, sigma, temperature_list, hws)
    with pm.Model() as model:
        theta = define_model_prior(number_free_parameters)
        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))
        print("calling the sampler")
        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.sample(
            num_samples,
            tune=num_tune,
            step=pm.Metropolis(),
            return_inferencedata=True,
        )

    save_folder = (
        "test_results2/num_samples="
        + str(num_samples)
        + " num_tune="
        + str(num_tune)
        + " sigma="
        + str(sigma)
        + " temperature_list="
        + str(len(temperature_list))
        + " number_free_parameters="
        + str(number_free_parameters)
    )
    os.makedirs(save_folder, exist_ok=True)

    plot_trace(idata_mh, true_parameters, save_folder, savefig=True)
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
    PL_results = data.D.kr_hw.reshape(-1, 1)

    return PL_results / max(PL_results)


def pl_loglike(theta, data, sigma, temperature_list, hws):
    if type(theta) == np.ndarray:
        theta = theta.tolist()
    # theta[0] = theta[0] * 10.0
    # theta[1] = theta[1] * 0.1
    model_data = pl_trial(list(theta), temperature_list, hws)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data).any() or np.isinf(model_data).any():
        return -np.inf
    return -(0.5 / sigma**2) * np.sum((data - model_data) ** 2)


def generate_data(temperature_list, hws,sigma):
    E_true = 1.5
    sigma_true = 0.02
    LI_true = 0.11
    L0_true = 0.1
    H0_true = 0.15
    truemodel_pl = pl_trial(
        [E_true, sigma_true, LI_true, L0_true, H0_true], temperature_list, hws
    ) + np.random.normal(0, sigma, size=(len(hws)*len(temperature_list), 1))
    true_parameters = [E_true, sigma_true, LI_true, L0_true, H0_true]
    return truemodel_pl, true_parameters


def define_model_prior(number_free_parameters):
    if number_free_parameters == 2:
        E = pm.Uniform("E", 1, 1.8)
        LI = pm.Uniform("LI", 0.06, 0.2)
        theta = pt.as_tensor_variable([E, LI])
    elif number_free_parameters == 5:
        E = pm.Uniform("E", 1.4, 1.6)
        sigma = pm.Uniform("sigma", 0.015, 0.025)
        LI = pm.Uniform("LI", 0.06, 0.2)
        L0 = pm.Uniform("L0", 0.06, 0.2)
        H0 = pm.Uniform("H0", 0.1, 0.2)
        theta = pt.as_tensor_variable([E, sigma, LI, L0, H0])
    return theta


def plot_trace(idata_mh, true_parameters, save_folder, savefig=True):
    axes = az.plot_trace(
        idata_mh,
        lines=[
            ("E", {}, true_parameters[0]),
            ("sigma", {}, true_parameters[1]),
            ("LI", {}, true_parameters[2]),
            ("L0", {}, true_parameters[3]),
            ("H0", {}, true_parameters[4]),
        ],
    )
    fig = axes.ravel()[0].figure
    fig.suptitle("Trace plot " + save_folder.split("/")[-1])
    if savefig:
        fig.savefig(save_folder + "/trace.png")


def plot_posterior_prediction(
    idata_mh, truemodel_pl, temperature_list, hws, save_folder, savefig=True
):
    theta_mean = idata_mh.posterior.mean(dim=["chain", "draw"])

    theta_mean_list = []
    for x in theta_mean.data_vars:
        theta_mean_list.append(
            theta_mean[x].values
        )  # theta_mean = theta_mean.to_dict()["values"]
    data_plot = pl_trial(theta_mean_list, temperature_list, hws)
    data_plot = data_plot.reshape(len(hws), -1)
    data_true_plot = truemodel_pl.reshape(len(hws), -1)
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
    ax.set_title("Posterior mean prediction")
    ax.legend()
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

    def __init__(self, loglike, data, sigma, temperature_list, hws):
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

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(
            theta, self.data, self.sigma, self.temperature_list, self.hws
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
        "--test_number", type=int, default=0, help="test number to run"
    )
    test_number = argparser.parse_args().test_number
    print(f"Running test number {test_number}")
    num_samples = [100, 3000, 5000]
    num_tune = [50, 1000]
    sigma = [0.02, 0.05]
    temperature_list = [
        [300.0, 250.0, 200.0, 150.0, 80.0],
        [300.0, 150.0, 80.0],
        [300.0],
    ]
    number_free_parameters = [2, 5]
    parameter_list = generate_parameter_list(
        num_samples, num_tune, sigma, temperature_list, number_free_parameters
    )

    # print the parameter list
    print(parameter_list[test_number])
    parameter_list[test_number]["test_number"] = test_number
    main(**parameter_list[test_number])  # run the example
