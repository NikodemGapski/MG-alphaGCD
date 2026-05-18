#!/bin/bash
#
#SBATCH --job-name=mg_agcd_build
#SBATCH --partition=a100
#SBATCH --qos=ngapski_a100
#SBATCH --gres=gpu:1
#SBATCH --output=std.out
#SBATCH --nodelist=a100a
#SBATCH --export=ALL

cmake -DNVSHMEM_MPI_SUPPORT=1 -DCMAKE_PREFIX_PATH="$HOME/libnvshmem-linux-x86_64-3.6.5_cuda12-archive/lib/cmake/nvshmem" -DCUDA_HOME="/usr/local/cuda" -S . -B build

cd build
make -j6