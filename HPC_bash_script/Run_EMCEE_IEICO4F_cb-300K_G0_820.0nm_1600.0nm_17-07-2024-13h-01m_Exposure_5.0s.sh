#!/bin/bash 
#PBS -l walltime=07:59:01 
#PBS -l select=1:ncpus=1:mem=80gb:avx=true 
 
cd /rds/general/user/hy2120/home/pl_temp_fit 
module load anaconda3/personal 
source activate pl_temp_fit     
python src/pl_temp_fit/scripts/new_pl_sampling_alllifetimes_abs.py --model_config_id 94ac833c-8019-4af9-805e-2cc449d055dd 
