#!/bin/bash
#SBATCH --job-name=shor_a100       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2               # number of gpus per node
#SBATCH --constraint=a100          # 40G
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=er4138@princeton.edu


module purge
module load anaconda3/2024.10
# module load cudatoolkit/12.4
# module load nvhpc/24.5 

conda activate /scratch/network/er4138/cudaq-env
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_btl='^openib'

/usr/bin/time -v srun python3 bernstein-vazirani.py 2 