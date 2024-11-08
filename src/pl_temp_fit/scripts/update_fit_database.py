import logging
from pathlib import Path

from pl_temp_fit import HPC_utils, config_utils


def main():
    script_to_run = "new_pl_sampling_alllifetimes_abs"
    new_path = "dat_hanbo_311024/csv/"
    new_path_save_folder = "fit_experimental_emcee_pl/fit_data/"
    path_database = "fit_experimental_emcee_pl/fit_data_base/allLifetimes/"
    config_folder = "fit_experimental_emcee_pl/config/"

    for test_id in get_all_json_files(path_database):
        model_config, model_config_save = config_utils.load_model_config(
            test_id, database_folder=path_database
        )
        csv_name = model_config_save["csv_name_pl"]
        logging.info(model_config_save["csv_name_pl"].split("/")[-1])
        model_config_save = config_utils.update_csv_name_pl(
            model_config_save,
            csv_name,
            new_path,
            new_path_save_folder,
            script_to_run,
            test_id,
        )
        config_utils.updata_model_config(
            test_id, path_database, model_config_save
        )
        HPC_utils.save_slurm_script(
            test_id, script_to_run, model_config_save, config_folder
        )


def get_all_json_files(path_database):
    all_files = Path(path_database).rglob("*.json")
    json_files = []
    for file in all_files:
        json_files.append(file.name.split(".")[0])
    return json_files

if __name__ == "__main__":
    main()
