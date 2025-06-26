#!/bin/bash
#SBATCH --job-name=pl_temp_fit
#SBATCH --output=/home/mazzouzi/pl_temp_fit/slurm_script/output/%A_%a.out
#SBATCH --error=/home/mazzouzi/pl_temp_fit/slurm_script/error/%A_%a.err
#SBATCH --ntasks 1 --cpus-per-task=32
#SBATCH --mem=30GB
#SBATCH --time=12:30:00
conda activate /home/mazzouzi/miniconda3/envs/pl_temp_fit
cd /home/mazzouzi/pl_temp_fit 
 srun python src/pl_temp_fit/scripts/new_pl_sampling_abs.py --model_config_id abs_only/def474da-906b-4032-89a0-d8eafd315c42 
