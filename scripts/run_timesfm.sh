#!/bin/bash
#SBATCH --account=rpp-xli135
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00
./home/zzx/projects/rpp-xli135/zzx/timesfm/experiments/extended_benchmarks/run_timesfm.py