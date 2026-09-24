#!/bin/bash
#SBATCH --job-name=EAAI2_budget32c
#SBATCH --partition=fat
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget32c_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget32c_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# 32³ 5-seed 确认（CPU 版，避开 3090 争用）。采样/去噪/柔度全走 CPU。
# 注：base-48 在 CPU 上采样较慢，5 seed × 30 例预计 ~4-6h。
python -u code/explore_budget_refinement.py \
  --data data/ood_32.npz --ckpt results/checkpoint_32_b48_e300.pt \
  --device cpu --n_samples 30 --base 48 --depth 2 \
  --seeds "${SEEDS:-12000,22000,32000,42000,52000}" --ddim_steps 20 \
  --out data/ood_budget_sweep_32_b48_5seed.csv
echo "32³ 5-seed 确认完成"
