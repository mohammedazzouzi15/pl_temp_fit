""" code to estimate the paramter of the model that reproduce the PL data using PYMC"""
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
    sigma=0.001,
    data_file="data.csv",
    Temp_std_err=2,
    hws_std_err=0.002,
    relative_intensity_std_error=0.1,
):
    number_free_parameters = 5
    Exp_data, temperature_list, hws = read_data(data_file)
    model_config = {
        "number_free_parameters": number_free_parameters,
        "sigma": sigma,
        "Temp_std_err": Temp_std_err,
        "hws_std_err": hws_std_err,
        "relative_intensity_std_error": relative_intensity_std_error,
    }
    X = {'temperature_list':temperature_list, 'hws':hws}
    print (f"size of hw is {hws.shape}")
    print (f"size of temperature_list is {temperature_list.shape}")
    date=datetime.datetime.now().strftime("%Y_%m_%d")
    # generate the data
    save_folder = (
        f"fit_experimental/{date}/{data_file.split('/')[-1]}/num_samples="
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
    truemodel_pl, true_parameters = generate_data(temperature_list, hws, **model_config)
    true_model_pl_list,variance,arg_max_variance,co_var_mat = plot_generated_data(truemodel_pl, temperature_list, hws, save_folder, model_config, savefig=True)
    #shift the variance to the maximum of the experimental data
    variance_data = co_var_mat.diagonal().reshape(hws.shape[0],-1).copy()#variance.copy()
    argmax_data = np.argmax(Exp_data, axis=0)
    for i in range(len(temperature_list)):
        variance_data[:, i] = np.roll(variance_data[:, i], argmax_data[i]-arg_max_variance[i])
    variance_data = variance_data+sigma
    np.fill_diagonal(co_var_mat, variance_data.reshape(-1,1))
    plot_data_with_variance(Exp_data, temperature_list, hws, variance_data, save_folder)

    ## initialise the model and run the fit
    model = PLPYMCModel.PLPYMCModel()
    for key, value in model_config.items():
        model.model_config[key] = value
    print(model.model_config)  
    model.sampler_config['step'] = "Metropolis(tune_interval =200)"    
    #model.sampler_config['step'] = "DEMetropolis(scaling=[0.1,0.01,0.1,0.1,0.1],tune_interval =200)"#"[pm.DEMetropolis([self.E,self.LI,self.sigma_E]),pm.DEMetropolis([self.L0,self.H0])]"  
    print(model.sampler_config)
    model.fit(X, Exp_data,co_var_mat = co_var_mat, draws=num_samples, tune=num_tune, chains=4, step=model.sampler_config['step'] , return_inferencedata=True)
    print(az.summary(model.idata))
    model.plot_trace(true_parameters=None, save_folder=save_folder, savefig=True)
    # save the data
    model.idata.to_netcdf(save_folder + "/idata.nc")
    model.plot_posterior_prediction(Exp_data,  temperature_list, hws, save_folder=save_folder, savefig=True)
    fname = "/model.nc"
    model.save(save_folder+fname)
    # plot the model data from the mean of the posterior   )
    return  model


def plot_data(
     truemodel_pl, temperature_list, hws,title="Experimental Data"
):
    fig,ax= plt.subplots(1,2, figsize=(10, 5))

    data_true_plot = truemodel_pl.reshape(len(hws), -1)/max(truemodel_pl.reshape(-1, 1))
    for i in range(len(temperature_list)):

        ax[0].plot(
            hws,
            data_true_plot[:, i],
            label="" + str(temperature_list[i]) + " K",
            #linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )

        ax[1].plot(
            hws,
            data_true_plot[:, i]/max(data_true_plot[:, i]),
            label="" + str(temperature_list[i]) + " K",
            #linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )
    ax[0].set_xlabel("Photon Energy (eV)")
    ax[0].set_ylabel("PL Intensity (arb. units)")
    ax[1].set_xlabel("Photon Energy (eV)")
    ax[1].set_ylabel("PL normalised (arb. units)")
    fig.suptitle(title)
    ax[0].legend(ncol=len(temperature_list), bbox_to_anchor=(1,-0.1
                                                             ), loc='upper center')
    return fig,ax


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


def plot_data_with_variance(
     Exp_data, temperature_list, hws, variance_data, save_folder,savefig=False):

    fig, axes = plt.subplots(1,len(temperature_list), figsize=(20, 5))

    for i in range(len(temperature_list)):
        ax=axes[i]
        ax.plot(
            hws,
            Exp_data[:, i],
            label="true" + str(temperature_list[i]) + " K",
            linestyle="--",
            color="C" + str(i),
            alpha=0.3,
        )
        ax.fill_between(
                hws,
                Exp_data[:, i]
                - np.sqrt(variance_data[:, i]),
                Exp_data[:, i]
                + np.sqrt(variance_data[:, i]),
                alpha=0.3,
                color="C" + str(i),
            )
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL Intensity (arb. units)")
        ax.set_title("temperature="+str(temperature_list[i])+" K")
        #ax.set_yscale('log')
        ax.set_ylim([1e-1, 1])
    fig.tight_layout()
    if savefig:
        fig.savefig(save_folder+"/data_with_variance.png")
    return fig,axes


#plot the generated data
def plot_generated_data(truemodel_pl, temperature_list, hws, save_folder, model_config, savefig=True,true_parameters=None):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    true_model_pl_list = []

    for x in range(100):
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
    arg_max_variance = np.argmax(mean_value_plot, axis=0)
    co_var = np.array(true_model_pl_list)
    co_var = co_var.reshape(100,-1)
    co_var_mat = np.cov(co_var[:,:].T)
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
    return mean_value_plot,variance,arg_max_variance,co_var_mat,true_model_pl_list


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
    num_samples, num_tune, sigma, 
    Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
):
    from itertools import product

    parameter_list = []
    parameter_list.append({
                "num_samples": 1000,
                "num_tune": 1000,
                "sigma": 0.001,
                "Temp_std_err": 5,
                "hws_std_err": 0.001,
                "relative_intensity_std_error": 0.01,
            })
    for i, j, k, l, m, n in product(
        num_samples,
        num_tune,
        sigma,
        Temp_std_err_list,
        hws_std_err_list,
        relative_intensity_std_error_list,
    ):
        parameter_list.append(
            {
                "num_samples": i,
                "num_tune": j,
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
    num_samples = [2000]
    num_tune = [2000]
    sigma = [0.01,0.002]

    Temp_std_err_list = [2,10]
    hws_std_err_list = [0.05,0.005]
    relative_intensity_std_error_list = [0.1,0.01]

    parameter_list = generate_parameter_list(
        num_samples, num_tune, sigma,
        Temp_std_err_list, hws_std_err_list, relative_intensity_std_error_list
    )

    # print the parameter list
    print(parameter_list[test_number])
    print(len(parameter_list))
    #parameter_list[test_number]["test_number"] = test_number
    main(data_file=argparser.parse_args().data_file,**parameter_list[test_number])  # run the example