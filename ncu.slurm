#!/bin/bash
#SBATCH --job-name=dj_ncu_a100       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G       # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100         # GPU type
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=er4138@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load cudatoolkit/12.4
module load anaconda3/2024.10
module load nvhpc/24.5 
module load cudatoolkit/12.6
module load openmpi/nvhpc-24.5/4.1.6

conda activate /scratch/network/er4138/cudaq-env

ncu --clock-control=none -o my_report_${dj_ncu_a100} python3 deutsch-jozsa.py 4

