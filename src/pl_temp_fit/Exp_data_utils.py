import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


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
    truemodel_pl = np.array(data[1:, 1:]).transpose()  # .reshape(-1, 1)
    return truemodel_pl, temperature_list, hws


def plot_pl_data_with_variance(
    Exp_data, temperature_list, hws, variance_data, save_folder, savefig=False
):
    fig, axes = plt.subplots(1, len(temperature_list), figsize=(20, 5))
    Exp_data = Exp_data / max(Exp_data.reshape(-1, 1))
    for i in range(len(temperature_list)):
        ax = axes[i]
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
            Exp_data[:, i] - np.sqrt(variance_data[:, i]),
            Exp_data[:, i] + np.sqrt(variance_data[:, i]),
            alpha=0.3,
            color="C" + str(i),
        )
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL Intensity (arb. units)")
        ax.set_title("temperature=" + str(temperature_list[i]) + " K")
        # ax.set_yscale('log')
        ax.set_ylim([0, 1])
    fig.tight_layout()
    if savefig:
        fig.savefig(save_folder + "/data_with_variance.png")
    return fig, axes


def plot_pl_data(
    truemodel_pl, temperature_list, hws, title="Experimental Data"
):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    data_true_plot = truemodel_pl.reshape(len(hws), -1) / max(
        truemodel_pl.reshape(-1, 1)
    )
    for i in range(len(temperature_list)):
        ax[0].plot(
            hws,
            data_true_plot[:, i],
            label="" + str(temperature_list[i]) + " K",
            # linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )

        ax[1].plot(
            hws,
            data_true_plot[:, i] / max(data_true_plot[:, i]),
            label="" + str(temperature_list[i]) + " K",
            # linestyle="--",
            color="C" + str(i),
            linewidth=2,
        )
    ax[0].set_xlabel("Photon Energy (eV)")
    ax[0].set_ylabel("PL Intensity (arb. units)")
    ax[1].set_xlabel("Photon Energy (eV)")
    ax[1].set_ylabel("PL normalised (arb. units)")
    fig.suptitle(title)
    ax[0].legend(
        ncol=len(temperature_list),
        bbox_to_anchor=(1, -0.1),
        loc="upper center",
    )
    return fig, ax


def change_wavelength_range(
    csv_name, hws_limits=[0.95, 1.6], step=0.01, temperature_list=None
):
    """Adjust the wavelength range of experimental data and splits the data based on temperature ranges.

    Parameters
    ----------
    csv_name (str or Path): The path to the CSV file containing the experimental data.
    hws_limits (list, optional): The lower and upper limits of the wavelength range. Defaults to [0.95, 1.6].
    step (float, optional): The step size for the wavelength range. Defaults to 0.01.
    temperature_split (list, optional): A list of tuples specifying the temperature ranges to split the data. Defaults to an empty list.

    Returns
    -------
    list: A list of paths to the new CSV files created after adjusting the wavelength range and splitting the data.

    """
    data, temperature_list_old, hws_old = read_data(csv_name)
    print(temperature_list_old)
    if temperature_list is not None:
        temperature_list = np.array(
            [temp for temp in temperature_list_old if temp in temperature_list]
        )
    else:
        temperature_list = temperature_list_old
    csv_names = []
    csv_name = csv_name.absolute().as_posix()

    list_i = [
        i
        for i in range(len(temperature_list_old))
        if temperature_list_old[i] in temperature_list
    ]
    hws = np.arange(hws_limits[0], hws_limits[1], step)
    y = np.zeros((hws.size, 1 + len(temperature_list)))
    y[:, 0] = hws
    for _j, _i in enumerate(list_i):
        f = interpolate.interp1d(
            hws_old, data[:, _i], axis=0, fill_value="extrapolate"
        )
        y[:, _j + 1] = f(hws)
    data_new = np.zeros((len(temperature_list) + 1, len(hws) + 1))
    data_new[:, 1:] = y.transpose()
    data_new[1:, 0] = temperature_list
    data_new = pd.DataFrame(data_new, columns=["Temperature"] + list(hws))
    new_csv_name = f'{csv_name.replace(".csv","_mod.csv")}'
    data_new.to_csv(
        new_csv_name,
        index=None,
        header=None,
    )

    return new_csv_name


def from_xslx_to_csv(xlsx_file):
    data = pd.read_excel(xlsx_file)
    hws = np.arange(0.95, 1.8, 0.01)
    y = np.zeros((hws.size, len(data.values[0, :])))
    y[:, 0] = hws
    for i in range(1, len(data.values[0, :])):
        f = interpolate.interp1d(
            data.values[1:, 0],
            data.values[1:, i],
            axis=0,
            fill_value="extrapolate",
        )
        y[:, i] = f(hws)
    data = pd.DataFrame(y, columns=data.columns, index=None)
    data = data.transpose()
    data = data.rename(columns=data.iloc[0])[1:]
    if data.shape[0] > 8:
        data = data.iloc[range(0, data.shape[0], 2), :]
    data.to_csv(f'{xlsx_file.replace(".xlsx",".csv")}')
    return f'{xlsx_file.replace(".xlsx",".csv")}'
