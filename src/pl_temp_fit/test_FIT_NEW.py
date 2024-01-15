# define a pytensor Op for our likelihood function
from pl_temp_fit import PLPYMCModel
import arviz as az
import pymc as pm
print(f"Running on PyMC v{pm.__version__}")
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

def main(
    num_samples=50,
    num_tune=10,
    sigma=0.1,
    temperature_list=np.array([300.0, 150.0, 80.0]),
    number_free_parameters=2,
    hws=np.arange(0.8, 2, 0.01),
    Temp_std_err=2,
    hws_std_err=0.002,
    relative_intensity_std_error=0.1,
):  
    model_config = {
            "number_free_parameters": number_free_parameters,
            "sigma": sigma,
            "Temp_std_err": Temp_std_err,
            "hws_std_err": hws_std_err,
            "relative_intensity_std_error": relative_intensity_std_error,
        }
    X = {'temperature_list':temperature_list, 'hws':hws}
    # generate the data
    true_parameters=[1.5, 0.02, 0.09, 0.1, 0.15]
    truemodel_pl, true_parameters = generate_data(temperature_list, hws, **model_config, true_parameters=true_parameters)
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_folder = (
        f"test_results_PL/{date}/num_samples="
        + str(num_samples)
        + " num_tune=" + str(num_tune)
        + " sigma=" + str(sigma)
        + " temperature_list=" + str(len(temperature_list))
        + " number_free_parameters=" + str(number_free_parameters)
        + " Temp_std_err="+str(Temp_std_err)
        + " hws_std_err="+str(hws_std_err)
        + " relative_intensity_std_error="+str(relative_intensity_std_error)
    )
    os.makedirs(save_folder, exist_ok=True)
    true_model_pl_list,variance,arg_max_variance = plot_generated_data(truemodel_pl, temperature_list, hws, save_folder, model_config, savefig=True, true_parameters=true_parameters)
    variance = variance+ sigma # add min sigma to the variance
    ## initialise the model and run the fit
    model = PLPYMCModel.PLPYMCModel()
    for key, value in model_config.items():
        model.model_config[key] = value
    print(model.model_config)      
    model.sampler_config['step'] = "pm.DEMetropolis()"#"[pm.DEMetropolis([self.E,self.LI,self.sigma_E]),pm.DEMetropolis([self.L0,self.H0])]"  
    print(model.sampler_config)
    model.fit(X, truemodel_pl,sigma = variance, draws=num_samples, tune=num_tune, chains=4, step=model.sampler_config['step'] , return_inferencedata=True)
    print(az.summary(model.idata))
    model.plot_trace(true_parameters, save_folder=save_folder, savefig=True)
    # save the data
    model.idata.to_netcdf(save_folder + "/idata.nc")
    model.plot_posterior_prediction(truemodel_pl,  temperature_list, hws, save_folder=save_folder, savefig=True)
    fname = "/model.nc"
    model.save(save_folder+fname)
    # plot the model data from the mean of the posterior
    return model


#plot the generated data
def plot_generated_data(truemodel_pl, temperature_list, hws, save_folder, model_config, savefig=True,true_parameters=None):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    true_model_pl_list = []

    for x in range(100):
        truemodel_pl, true_parameters = generate_data(temperature_list, hws,**model_config, true_parameters=true_parameters)
        data_true_plot = truemodel_pl.reshape(len(hws), -1)
        data_true_plot = data_true_plot/max(data_true_plot.reshape(-1, 1))
        true_model_pl_list.append(data_true_plot.reshape(len(hws), len(temperature_list)))

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
    arg_max_variance = np.argmax(mean_value_plot, axis=0)

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
    return mean_value_plot,variance,arg_max_variance


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
    truemodel_pl = PLPYMCModel.pl_trial(
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


# generate a function to go through the parameter of the main function
# and generate a list of parameters to be passed to the main function
def generate_parameter_list(
    num_samples, num_tune, sigma, temperature_list, number_free_parameters,
    Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
):
    from itertools import product

    parameter_list = []
    parameter_list.append({
                "num_samples": 500,
                "num_tune": 100,
                "sigma": 0.01,
                "temperature_list": np.array([300.0, 150.0, 80.0]),
                "number_free_parameters": 2,
                "Temp_std_err": 5,
                "hws_std_err": 0.001,
                "relative_intensity_std_error": 0.01,
            })
    for i, j, k, l, m, n, o, p  in product(
        num_samples,
        num_tune,
        sigma,
        temperature_list,
        number_free_parameters,
        Temp_std_err_list,
        hws_std_err_list,
        relative_intensity_std_error_list,
    ):
        parameter_list.append(
            {
                "num_samples": i,
                "num_tune": j,
                "sigma": k,
                "temperature_list": np.array(l),
                "number_free_parameters": m,
                "Temp_std_err": n,
                "hws_std_err": o,
                "relative_intensity_std_error": p,
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
    num_samples = [4000]
    num_tune = [2000]
    sigma = [0.01]

    temperature_list = [
        [300.0, 250.0, 200.0, 150.0, 80.0],
        [300.0, 150.0, 80.0],
    ]

    number_free_parameters = [5]
    Temp_std_err_list = [2,10,0.1]
    hws_std_err_list = [0.005,0.001]
    relative_intensity_std_error_list = [0.05,0.01]

    parameter_list = generate_parameter_list(
        num_samples, num_tune, sigma, temperature_list, number_free_parameters,
        Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
    )

    # print the parameter list
    print(parameter_list[test_number])
    print(len(parameter_list))
    #parameter_list[test_number]["test_number"] = test_number
    main(**parameter_list[test_number])  # run the example
