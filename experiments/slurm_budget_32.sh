#!/bin/bash
#SBATCH --job-name=EAAI2_budget32
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget32_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget32_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 32³ 留出集上做「体积重标定 vs SIMP 参考」扫描（采样走 GPU，FEM 用 CG 求解器）
# 注：32³ 下 spsolve(LU) 因 3D 带宽过大超时；改 CG（不收敛但柔度已稳定）。
# 先单 seed × 30 例拿信号；后续再扩到 5 seed。
python -u code/explore_budget_refinement.py \
  --data data/ood_32.npz --ckpt "${CKPT:-results/checkpoint_32.pt}" \
  --device cuda --n_samples 30 --base "${BASE:-32}" --depth "${DEPTH:-2}" \
  --seeds ${SEEDS:-12000} --ddim_steps 20 \
  --out "${OUT:-data/ood_budget_sweep_32.csv}"
echo "32³ 体积重标定扫描完成"
