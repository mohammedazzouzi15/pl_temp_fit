from pymc_experimental.model_builder import ModelBuilder
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pl_temp_fit import LTPL
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pymc.util import RandomState
import arviz as az
from pathlib import Path
import json
import matplotlib.pyplot as plt


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
    PL_results = data.D.kr_hw#.reshape(-1, 1)

    return PL_results# / max(PL_results)


class PLPYMCModel(ModelBuilder):
    # Give the model a name
    _model_type = "PLFitModel"

    # And a version
    version = "0.1"

    def build_model(self, X, y, sigma = 0.1, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : Dict with the following keys:
            temperature_list : 
            hws :
        y : 
            The target data for the model. 
        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        temperature_list=X['temperature_list']
        truemodel_pl = y
        hws=X['hws']
        logl = LogLike(self.pl_loglike, truemodel_pl, sigma, temperature_list, hws)
        with pm.Model() as self.model:
            theta = self.define_model_prior()

            # use a Potential to "call" the Op and include it in the logp computation
            pm.Potential("likelihood", logl(theta))
       
    def pl_loglike(self, theta, data, sigma, temperature_list, hws):
        if type(theta) == np.ndarray:
            theta = theta.tolist()
        data = data/np.max(data.reshape(-1, 1))
        model_data = pl_trial(list(theta), temperature_list, hws)
        model_data = model_data/np.max(model_data.reshape(-1, 1))
        if type(sigma) is not np.ndarray:
            sigma = sigma * np.ones_like(model_data)
        assert model_data.shape == sigma.shape
        # check that the data in model_data does not contain NaNs or infs
        if np.isnan(model_data).any() or np.isinf(model_data).any():
            return -np.inf
        return -(0.5) * np.sum(((data - model_data)/sigma) ** 2)

    def define_model_prior(self):
        if self.model_config['number_free_parameters']== 2:
            E = pm.Uniform("E", self.model_config["E"]["min"], self.model_config["E"]["max"])
            LI = pm.TruncatedNormal('LI', mu=self.model_config["LI"]["mu"], sigma=self.model_config["LI"]["sigma"],
                                     lower=self.model_config["LI"]["lower"],upper=self.model_config["LI"]["upper"])
            theta = pt.as_tensor_variable([E, LI])
        elif self.model_config['number_free_parameters'] == 5:
            E = pm.Uniform("E", self.model_config["E"]["min"], self.model_config["E"]["max"])
            sigma_E = pm.Uniform("sigma_E", self.model_config["sigma_E"]["min"], self.model_config["sigma_E"]["max"])
            LI = pm.TruncatedNormal('LI', mu=self.model_config["LI"]["mu"], sigma=self.model_config["LI"]["sigma"],
                                        lower=self.model_config["LI"]["lower"],upper=self.model_config["LI"]["upper"])
            L0 = pm.TruncatedNormal('L0', mu=self.model_config["L0"]["mu"], sigma=self.model_config["L0"]["sigma"],
                                        lower=self.model_config["L0"]["lower"],upper=self.model_config["L0"]["upper"])
            H0 = pm.TruncatedNormal('H0', mu=self.model_config["H0"]["mu"], sigma=self.model_config["H0"]["sigma"],
                                        lower=self.model_config["H0"]["lower"],upper=self.model_config["H0"]["upper"])

            theta = pt.as_tensor_variable([E, sigma_E, LI, L0, H0])
        return theta
    
    def sample_model(self, **kwargs):
        """
        Sample from the PyMC model.

        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )

        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            
            step = sampler_args['step']
            sampler_args.pop('step')
            print(sampler_args)
            idata = pm.sample(step=eval(step),**sampler_args)
            idata.extend(pm.sample_prior_predictive(), join="right")
            idata.extend(pm.sample_posterior_predictive(idata), join="right")

        idata = self.set_idata_attrs(idata)
        return idata


    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = {
            "number_free_parameters": 2,
            "E": {"min": 1.0, "max": 2.0},
            "sigma_E": {"min": 0.001, "max": 0.03},
            "LI": {"mu": 0.12, "sigma": 0.01, "lower": 0.05, "upper": 0.15},
            "L0": {"mu": 0.12, "sigma": 0.01, "lower": 0.05, "upper": 0.15},
            "H0": {"mu": 0.12, "sigma": 0.01, "lower": 0.12, "upper": 0.18},
            "Temp_std_err": 2,
            "hws_std_err": 0.002,
            "relative_intensity_std_error": 0.1,
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 10,
            "tune": 5,
            "chains": 3,
            "step": "pm.Nuts()",
            "return_inferencedata":True,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass
    
    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        self.model_coords = None  # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y
    

    def fit(
        self,
        X: Dict,
        y: Optional[pd.Series] = None,
        sigma: Optional[np.ndarray] = None,
        progressbar: bool = True,
        predictor_names: List[str] = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : array-like if sklearn is available, otherwise array, shape (n_obs, n_features)
            The training input samples.
        y : array-like if sklearn is available, otherwise array, shape (n_obs,)
            The target values (real numbers).
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        predictor_names: List[str] = None,
            Allows for custom naming of predictors given in a form of 2dArray
            allows for naming of predictors when given in a form of np.ndarray, if not provided the predictors will be named like predictor1, predictor2...
        random_seed : RandomState
            Provides sampler with initial random seed for obtaining reproducible samples
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            returns inference data of the fitted model.
        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(data)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        """
        if predictor_names is None:
            predictor_names = []
        self._generate_and_preprocess_model_data(X, y)
        if sigma is None:
            self.sigma = 0.1
        else:
            self.sigma = sigma
        self.build_model(self.X, self.y, self.sigma)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)


        return self.idata  # type: ignore

    @classmethod
    def load(cls, fname: str):
        """
        Creates a ModelBuilder instance from a file,
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of ModelBuilder.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.
        Examples
        --------
        >>> class MyModel(ModelBuilder):
        >>>     ...
        >>> name = './mymodel.nc'
        >>> imported_model = MyModel.load(name)
        """
        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(json.loads(idata.attrs["model_config"]))
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata

        # All previously used data is in idata.

        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model

    def plot_trace(self, true_parameters, save_folder, savefig=True):
        axes = az.plot_trace(
            self.idata,
            lines=[
                ("E", {}, true_parameters[0]),
                ("sigma_E", {}, true_parameters[1]),
                ("LI", {}, true_parameters[2]),
                ("L0", {}, true_parameters[3]),
                ("H0", {}, true_parameters[4]),
            ],
        )
        fig = axes.ravel()[0].figure
        fig.suptitle("Trace plot " + save_folder.split("/")[-1])
        fig.tight_layout()
        if savefig:
            fig.savefig(save_folder + "/trace.png")
        return fig

    def plot_posterior_prediction(
        self, truemodel_pl, temperature_list, hws, save_folder, savefig=True
    ):
        theta_mean = self.idata.posterior.mean(dim=["chain", "draw"])

        theta_mean_list = []
        for x in theta_mean.data_vars:
            theta_mean_list.append(
                theta_mean[x].values
            )  # theta_mean = theta_mean.to_dict()["values"]
        data_plot = pl_trial(theta_mean_list, temperature_list, hws)
        data_plot = data_plot.reshape(len(hws), -1)
        data_plot = data_plot / max(data_plot.reshape(-1, 1))
        data_true_plot = truemodel_pl.reshape(len(hws), -1)
        data_true_plot = data_true_plot / max(data_true_plot.reshape(-1, 1))
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
        fig.tight_layout()
        if savefig:
            fig.savefig(save_folder + "/posterior_mean.png")
        return fig