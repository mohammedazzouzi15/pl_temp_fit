#!/bin/bash 
#PBS -l walltime=70:59:01 
#PBS -l select=1:ncpus=128:mem=80gb:avx=true 
 
cd /rds/general/user/hy2120/home/pl_temp_fit 
module load anaconda3/personal 
source activate pl_temp_fit     
python src/pl_temp_fit/scripts/new_pl_sampling_alllifetimes_abs.py --model_config_id 20e30f8a-fad4-458f-90ee-a2a2a7a29b8b 
