#!/bin/bash 
#PBS -l walltime=70:59:01 
#PBS -l select=1:ncpus=128:mem=80gb:avx=true 
 
cd /rds/general/user/hy2120/home/pl_temp_fit 
module load anaconda3/personal 
source activate pl_temp_fit     
python src/pl_temp_fit/scripts/new_pl_sampling_abs.py --model_config_id 682d4147-0238-4fc6-b5d4-15cb6e69db3f 
