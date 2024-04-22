#!/bin/bash 
#SBATCH --job-name=demo_user                     # job name
#SBATCH --account=gts-phentenryck3-ai4opt    # account to which job is charged
#SBATCH -N1 --ntasks-per-node=30                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16gb                    # Memory per cpu
#SBATCH -t2-00:00:00                               # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                          # Partition name (where job is submitted)
#SBATCH --output=/storage/home/hcoda1/0/tye63/p-phentenryck3-1/simopt/experiments/logs/demo_user-%a.out  # Combined output and error messages file
#SBATCH --array=3        # Array range

/storage/coda1/p-phentenryck3/1/tye63/miniconda3/envs/simopt/bin/python /storage/home/hcoda1/0/tye63/p-phentenryck3-1/simopt/demo/demo_user.py
