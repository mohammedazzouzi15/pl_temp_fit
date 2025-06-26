from pathlib import Path


def save_slurm_script(
    test_id, script_to_run, model_config_save, config_folder, sh_name=None
):
    model_config_id = config_folder + "/" + test_id
    csv_name = model_config_save["csv_name_pl"]
    results_name = csv_name.replace(".csv", "").split("/")[-1]
    if sh_name is None:
        sh_name = f"{results_name}_{script_to_run}.sh"
    script = """#!/bin/bash
#SBATCH --job-name=pl_temp_fit
#SBATCH --output=/home/mazzouzi/pl_temp_fit/slurm_script/output/%A_%a.out
#SBATCH --error=/home/mazzouzi/pl_temp_fit/slurm_script/error/%A_%a.err
#SBATCH --ntasks 1 --cpus-per-task=32
#SBATCH --mem=30GB
#SBATCH --time=12:30:00
conda activate /home/mazzouzi/miniconda3/envs/pl_temp_fit
cd /home/mazzouzi/pl_temp_fit"""
    script = (
        script
        + " \n "
        + f"srun python src/pl_temp_fit/scripts/{script_to_run}.py --model_config_id {model_config_id} \n"
    )
    print(sh_name)
    path_script = Path("slurm_script" + "/" + config_folder)
    path_script.mkdir(parents=True, exist_ok=True)
    file_script = path_script / f"{sh_name}"
    with file_script.open("wb") as f:
        f.write(bytes(script, "utf-8"))
