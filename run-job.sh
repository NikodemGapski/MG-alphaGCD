#!/bin/bash
#
#SBATCH --job-name=mg_agcd
#SBATCH --partition=a100
#SBATCH --qos=ngapski_a100
#SBATCH --gres=gpu:1
#SBATCH --output=std.out
#SBATCH --nodelist=a100a
#SBATCH --export=ALL

input="$(pwd)/../datasets/data/wiki-Vote/wiki-Vote.mtx"
cd build

mpirun -np 1 ./MG_GCD -path_of_graph $input