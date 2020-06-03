#!/bin/sh
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1

srun --unbuffered python /work/henriksv/code/mario-data-collection/mario_dqn/plots/compare_qlt.py