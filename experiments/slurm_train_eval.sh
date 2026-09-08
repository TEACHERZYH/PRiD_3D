#!/bin/bash
#SBATCH --job-name=EAAI2_train_eval
#SBATCH --partition=fat
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train_eval_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train_eval_%j.err

# EAAI2 完整流水线：训练 + 评估（自动串联）
set -e

source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== 1. 训练 ==="
python code/train.py \
    --data data/train_16.npz \
    --epochs 120 \
    --batch 8 \
    --device cpu \
    --timesteps 200 \
    --base 32 \
    --depth 2 \
    --out results/checkpoint_16.pt

echo "=== 2. 评估 ==="
python code/evaluate.py \
    --data data/train_16.npz \
    --ckpt results/checkpoint_16.pt \
    --out results/eval \
    --n_samples 30

echo "=== 流水线完成 ==="
ls -la results/ results/eval/
