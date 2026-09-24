#!/bin/bash
#SBATCH --job-name=EAAI2_train32
#SBATCH --partition=3090
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train32_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train32_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 32³ 模型训练（与 16³ 同架构：in_channels=5, base=32, depth=2）
python -u code/train.py \
  --data data/train_32.npz --epochs 120 --lr 1e-3 --batch 8 \
  --device cuda --timesteps 200 --base 32 --depth 2 \
  --out results/checkpoint_32.pt
echo "32³ 训练完成"
ls -la results/checkpoint_32.pt
