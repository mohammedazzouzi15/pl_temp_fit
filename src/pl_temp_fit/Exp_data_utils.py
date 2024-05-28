import numpy as np
import matplotlib.pyplot as plt


def read_data(csv_file):
    # read csv data where the first row is the photon energy
    # and the columns are the temperature
    # the first column is temperature

    # read the data
    data = np.genfromtxt(csv_file, delimiter=",")
    # the first row is the photon energy
    hws_PL = np.array(data[0, 1:])
    # the first column is temperature
    temperature_list_PL = np.array(data[1:, 0])
    # the rest of the data is the PL intensity
    truemodel_pl = np.array(data[1:, 1:]).transpose()#.reshape(-1, 1)
    return truemodel_pl, temperature_list_PL, hws_PL


def plot_PL_data_with_variance(
     Exp_data, temperature_list_PL, hws_PL, variance_data, save_folder,savefig=False):

    fig, axes = plt.subplots(1,len(temperature_list_PL), figsize=(20, 5))
    Exp_data = Exp_data/max(Exp_data.reshape(-1, 1)) 
    for i in range(len(temperature_list_PL)):
        ax=axes[i]
        ax.plot(
            hws_PL,
            Exp_data[:, i],
            label="true" + str(temperature_list_PL[i]) + " K",
            linestyle="--",
            color="C" + str(i),
            alpha=0.3,
        )
        ax.fill_between(
                hws_PL,
                Exp_data[:, i]
                - np.sqrt(variance_data[:, i]),
                Exp_data[:, i]
                + np.sqrt(variance_data[:, i]),
                alpha=0.3,
                color="C" + str(i),
            )
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL Intensity (arb. units)")
        ax.set_title("temperature="+str(temperature_list_PL[i])+" K")
        #ax.set_yscale('log')
        ax.set_ylim([0, 1])
    fig.tight_layout()
    if savefig:
        fig.savefig(save_folder+"/data_with_variance.png")
    return fig,axes

def plot_PL_data(
     truemodel_pl, temperature_list_PL, hws_PL,title="Experimental Data"
):
    fig,ax= plt.subplots(1,2, figsize=(10, 5))

    data_true_plot = truemodel_pl.reshape(len(hws_PL), -1)/max(truemodel_pl.reshape(-1, 1))
    for i in range(len(temperature_list_PL)):

        ax[0].plot(
            hws_PL,
            data_true_plot[:, i],
            label="" + str(temperature_list_PL[i]) + " K",
            #linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )

        ax[1].plot(
            hws_PL,
            data_true_plot[:, i]/max(data_true_plot[:, i]),
            label="" + str(temperature_list_PL[i]) + " K",
            #linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )
    ax[0].set_xlabel("Photon Energy (eV)")
    ax[0].set_ylabel("PL Intensity (arb. units)")
    ax[1].set_xlabel("Photon Energy (eV)")
    ax[1].set_ylabel("PL normalised (arb. units)")
    fig.suptitle(title)
    ax[0].legend(ncol=len(temperature_list_PL), bbox_to_anchor=(1,-0.1
                                                             ), loc='upper center')
    return fig,ax