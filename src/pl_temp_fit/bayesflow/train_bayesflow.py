import logging
from functools import partial
from pathlib import Path

import bayesflow.diagnostics as diag
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer

from pl_temp_fit import (
    Exp_data_utils,
    config_utils,
)
from pl_temp_fit.data_generators import PLAbsandAlllifetime

logging.basicConfig(level=logging.INFO)


model_config_id = "225e5b67-23f6-4435-9478-029693d41c3a"
model_config, model_config_save = config_utils.load_model_config(
    model_config_id,
    database_folder="/media/mohammed/Work/pl_temp_fit/fit_experimental_emcee_pl/fit_data_base/sensitivity_2/",
)
csv_name_pl = model_config_save["csv_name_pl"]
save_folder = model_config_save["save_folder"]
Path(save_folder).mkdir(parents=True, exist_ok=True)
# Load the data
exp_data_pl, temperature_list_pl, hws_pl = Exp_data_utils.read_data(
    "/media/mohammed/Work/pl_temp_fit/" + csv_name_pl
)
temperature_list_pl = [60, 150, 300]
# initialising the data generator
pl_data_gen = PLAbsandAlllifetime.PLAbsandAlllifetime(
    temperature_list_pl, hws_pl
)
pl_data_gen.relative_error_lifetime = model_config_save[
    "relative_error_lifetime"
]
pl_data_gen.error_in_max_abs_pos = model_config_save["error_in_max_abs_pos"]
pl_data_gen.max_abs_pos_exp = model_config_save["max_abs_pos_exp"]
pl_data_gen.temperature_lifetimes_exp = model_config_save[
    "temperature_lifetimes_exp"
]
pl_data_gen.update_with_model_config(model_config_save)


def simulator_of_spectra(params):
    """Simulates data for given parameters."""
    params_dict = {}
    for param_key in ["EX", "CT", "D"]:
        params_dict[param_key] = {}
        if pl_data_gen.min_bounds[param_key] == {}:
            continue

        for id, key in enumerate(pl_data_gen.min_bounds[param_key].keys()):
            params_dict[param_key][key] = params[id]

    return pl_data_gen.generate_data(params_dict)[0]


simulator = Simulator(simulator_fun=simulator_of_spectra)


def get_params_names(pl_data_gen):
    """Returns the names of the parameters."""
    counter = 0
    param_names = []
    for param_key in ["EX", "CT", "D"]:
        if pl_data_gen.min_bounds[param_key] == {}:
            continue

        for id, key in enumerate(pl_data_gen.min_bounds[param_key].keys()):
            param_names.append(key)
            counter += 1
    return param_names


def model_prior(pl_data_gen):
    """Generates random draws from uniform prior with specific ranges for each parameter."""
    counter = 0
    sample = []
    for param_key in ["EX", "CT", "D"]:
        if pl_data_gen.min_bounds[param_key] == {}:
            continue

        for id, key in enumerate(pl_data_gen.min_bounds[param_key].keys()):
            sample.append(
                np.random.uniform(
                    low=pl_data_gen.min_bounds[param_key][key],
                    high=pl_data_gen.max_bounds[param_key][key],
                )
            )
    return np.array(sample)


prior = Prior(
    prior_fun=partial(model_prior, pl_data_gen),
    param_names=get_params_names(pl_data_gen),
)
prior_means, prior_stds = prior.estimate_means_and_stds()

print(f"Prior means: {prior_means}")
print(f"Prior stds: {prior_stds}")
model = GenerativeModel(prior, simulator, name="PL spectra simulator")

sim_data = model(8)["sim_data"]
fig, axarr = plt.subplots(2, 4, figsize=(16, 6))
ax = axarr.flat

for i, data in enumerate(sim_data):
    for j, temperature in enumerate(pl_data_gen.temperature_list):
        ax[i].plot(hws_pl, data[:, j], label=f"{temperature} K")
    ax[i].set_xlabel("Photon Energy (eV)")
    ax[i].set_ylabel("PL Intensity (arb. units)")

    ax[i].grid(alpha=0.5)
    ax[i].legend()
    ax[i].set_title(f"Simulation #{i+1}")

plt.tight_layout()
plt.savefig("simulated_data.png")

class CustomConvNet(tf.keras.Model):
    def __init__(self,summary_output_size=32,hiddens=[128,128]):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(hiddens[0], kernel_size=100, padding='same', activation='tanh')
        self.conv2 = tf.keras.layers.Conv1D(hiddens[0], kernel_size=1, padding='same', activation='tanh')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hiddens[1], activation='tanh')
        self.dense2 = tf.keras.layers.Dense(summary_output_size)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

summary_net = CustomConvNet()
COUPLING_NET_SETTINGS = {
    "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout": False,
}
num_params = prior_means.shape[1]
inference_net = InvertibleNetwork(
    num_params=num_params,
    num_coupling_layers=5,
    coupling_settings=COUPLING_NET_SETTINGS,
)

amortizer = AmortizedPosterior(
    inference_net, summary_net, name="PL_spectra_amortizer"
)

trainer = Trainer(
    amortizer=amortizer,
    generative_model=model,
    memory=True,
    checkpoint_path="checkpoints.ckpt",
)
print(amortizer.summary())

history = trainer.train_online(
    epochs=30, iterations_per_epoch=500, batch_size=16
)

fig = diag.plot_losses(history)
fig.savefig("losses.png")
