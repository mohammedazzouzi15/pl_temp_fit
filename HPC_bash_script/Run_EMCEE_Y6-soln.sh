#!/bin/bash 
#PBS -l walltime=70:59:01 
#PBS -l select=1:ncpus=128:mem=80gb:avx=true 
 
cd /rds/general/user/hy2120/home/pl_temp_fit 
module load anaconda3/personal 
source activate pl_temp_fit     
python src/pl_temp_fit/scripts/new_pl_sampling_abs.py --model_config_id 7cee67c6-515f-49a5-8b8d-431ddbaeef11 
