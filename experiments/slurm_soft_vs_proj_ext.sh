#!/bin/bash
#SBATCH --job-name=EAAI2_soft_ext
#SBATCH --partition=fat
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/soft_ext_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/soft_ext_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 增补实验 1：软约束（热代理引导）对照 —— 三 seed 重跑
#   修复的缺陷：论文 §5.5 的「热代理 0.9510 vs 无条件 0.9511，p=0.97」
#   原为单 seed 归档结果，本机不可重算。此处用 3 个 seed × 30 样本配对重做，
#   产出可归档的逐样本 CSV（每行含 uncond/soft/refine 三条件的指标）。
for s in 12000 22000 32000; do
  echo "=== seed ${s} ==="
  python -u code/soft_constraint_baseline.py \
    --data data/train_16.npz --ckpt results/checkpoint_16.pt \
    --n_samples 30 --seed "${s}" \
    --guide_steps 120 --guide_lr 0.7 --guide_iters 60 --vol_weight 0.5 \
    --out "data/soft_vs_proj_seed${s}.csv"
done
echo "软约束对照（3 seed）完成"
