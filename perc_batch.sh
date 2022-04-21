#!/bin/bash
#
#SBATCH --job-name=percolate2
#SBATCH --partition=ivm
#SBATCH --nodelist node242
#SBATCH --ntasks=1
#SBATCH --time=01-00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5000
#SBATCH --output=out_net2
#SBATCH --error=err_net2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elco.koks@vu.nl

source activate py38
python /scistor/ivm/eks510/projects/trails/src/trails/network.py