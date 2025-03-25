#!/bin/bash
#SBATCH --account=rrg-timsbc
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G              # memory per node
#SBATCH --time=0-19:00
source /home/zzx/projects/rrg-timsbc/zzx/bin/activate
module load python/3.11
module load scipy-stack
module load gcc arrow/18.1.0
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
# pip install seaborn
#pip install --no-index -r /home/zzx/projects/rrg-timsbc/zzx/timesfm/scripts/requirements.txt
echo "Running "
python /home/zzx/projects/rrg-timsbc/zzx/timesfm/src/finetuning/finetuning_example_weather.py --training_mode=single
# python /home/zzx/projects/rrg-timsbc/zzx/timesfm/src/finetuning/finetuning_example_naive_weather.py --training_mode=single