#!/bin/bash
#SBATCH --job-name=EAAI2_ood32
#SBATCH --partition=fat
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood32_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood32_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 32³ 留出集（三类分布偏移各 10 例：volfrac/domain/load）
python -u code/data_generation/synthesize_ood.py \
  --nx 32 --n_samples 30 --out data/ood_32.npz
echo "32³ OOD 留出集合成完成"
ls -la data/ood_32.npz
