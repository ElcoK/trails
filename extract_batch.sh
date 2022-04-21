#!/bin/bash
#
#SBATCH --job-name=get_pop_od
#SBATCH --partition=ivm
#SBATCH --nodelist node240
#SBATCH --ntasks=1
#SBATCH --time=04-00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=25000
#SBATCH --output=out_extract
#SBATCH --error=err_extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elco.koks@vu.nl

source activate py38
python3 /scistor/ivm/eks510/projects/trails/src/trails/simplify_all.py