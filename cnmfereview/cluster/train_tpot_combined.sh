#!/bin/bash
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00 # D-HH:MM
#SBATCH --output=%N-%j.out


SOURCEDIR=~/cnmfereview

# activate environment
module load python/3.6
source ~/sklearn/bin/activate

# prepare data, usually only if using a dataloader (too many i/o calls)
mkdir $SLURM_TMPDIR/data
tar -xf ~/data/cnmfereview.tar -C $SLURM_TMPDIR/data

# TODO identify and run script
SCRIPT=$SOURCEDIR/cnmfereview/ml/train_tpot.py

python $SCRIPT -t 1 -f 'combined' -d $SLURM_TMPDIR -s $SOURCEDIR