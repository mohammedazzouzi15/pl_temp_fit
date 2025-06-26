# load the relevant modules for the analysis

import os
from pathlib import Path

import emcee
import numpy as np
import pandas as pd

from pl_temp_fit import config_utils


def main(name_folder="sensitivity"):
    databse_path = Path(
        f"fit_experimental_emcee_pl/fit_data_base/{name_folder}/"
    )

    # databse_path = Path(
    #    "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/fit_experimental_emcee_pl/fit_data_base/allLifetimes/"
    # )

    add_for_ssh = ""  # "/run/user/1000/gvfs/sftp:host=lcmdlc3.epfl.ch,user=mazzouzi/home/mazzouzi/pl_temp_fit/"  # ""
    json_files = list(databse_path.glob("*.json"))
    list_model_config = []
    for _id, json_file in enumerate(json_files):
        model_config, model_config_save = config_utils.load_model_config(
            json_file.name.replace(".json", ""),
            database_folder=databse_path,
        )
        if os.path.exists(
            add_for_ssh + model_config_save["save_folder"] + "/sampler.h5"
        ):
            filename = (
                add_for_ssh + model_config_save["save_folder"] + "/sampler.h5"
            )
            try:
                reader = emcee.backends.HDFBackend(filename, name="multi_core")
                if not reader.initialized:
                    print("multi_core empty file")
                    continue
                if reader.iteration == 0:
                    print("empty file")
                    continue
                distribution = reader.get_chain(flat=True)
            except AttributeError:
                print("check if single core")

                reader = emcee.backends.HDFBackend(
                    filename, name="single_core"
                )
                if not reader.initialized:
                    print("empty file")
                    continue
                distribution = reader.get_chain(flat=True)

            true_parameters = list(np.mean(distribution, axis=0))
            model_config_save["mean"] = [f"{x:.3f}" for x in true_parameters]
            model_config_save["num_iteration"] = reader.iteration
            log_prob = reader.get_log_prob(flat=True)
            model_config_save["max_log_prob"] = np.max(log_prob)
            list_model_config.append(model_config_save)
    if len(list_model_config) == 0:
        print("no data")
    else:
        df_all = pd.DataFrame(list_model_config)
        df_all.sort_values(by="date", ascending=False, inplace=True)
    df_all["csv_name_pl"] = df_all["csv_name_pl"].apply(
        lambda x: x.split("/")[-1]
    )
    df_all.to_csv(f"script/all_results{name_folder}.csv", index=False)


if __name__ == "__main__":
    main()
