#!/bin/bash
#SBATCH --job-name=QAOA_a100       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core 
#SBATCH --gres=gpu:1               # number of gpus per node
#SBATCH --constraint=a100          # 40G
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=er4138@princeton.edu


module purge
module load anaconda3/2024.10

conda activate /scratch/network/er4138/cudaq-env
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_btl='^openib'


/usr/bin/time -v python3 QAOA.py 29
/usr/bin/time -v python3 QAOA.py 29
