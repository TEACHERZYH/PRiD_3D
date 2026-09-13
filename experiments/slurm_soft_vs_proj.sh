#!/bin/bash
#SBATCH --job-name=EAAI2_soft_vs_proj
#SBATCH --partition=fat
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/soft_vs_proj_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/soft_vs_proj_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# P0 实验（修正版）：界定（投影精炼）vs 拉扯（软约束引导，等体积 + 半隐式热扩散）
python -u code/soft_constraint_baseline.py --n_samples 30 --guide_steps 120 --guide_lr 0.7 --guide_iters 60 --vol_weight 0.5 --out data/soft_vs_proj_v2.csv
echo "P0 修正版对照完成"
