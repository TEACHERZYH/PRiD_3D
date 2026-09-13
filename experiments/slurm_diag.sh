#!/bin/bash
#SBATCH --job-name=EAAI2_diag
#SBATCH --partition=fat
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/diag_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/diag_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# 诊断：原 checkpoint_16.pt（作为 clean 行） vs 本次新训的 checkpoint_16_clean.pt（作为 dirty 行）
#   目的：确认"新训 clean 模型采样连通性 0.7533"是模型问题还是评估问题
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_16.pt --dirty results/checkpoint_16_clean.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/diag_orig_vs_newclean.csv
echo "诊断完成"
