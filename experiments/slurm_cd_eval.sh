#!/bin/bash
#SBATCH --job-name=EAAI2_cd_eval
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cd_eval_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cd_eval_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# clean vs dirty 对照评估（模型已训练完成，仅评估）
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_16_clean.pt --dirty results/checkpoint_16_dirty.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/clean_vs_dirty.csv
echo "对照评估完成"
