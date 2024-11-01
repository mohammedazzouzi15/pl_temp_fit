#!/bin/bash

# Directory containing the scripts
SCRIPT_DIR="/home/mazzouzi/pl_temp_fit/slurm_script/all_temp"

# Loop through all files in the directory
for script in "$SCRIPT_DIR"/*
do
        sbatch "$script"
done