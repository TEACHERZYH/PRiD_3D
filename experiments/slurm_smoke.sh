#!/bin/bash
#SBATCH --job-name=EAAI2_smoke
#SBATCH --partition=3090
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/smoke_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/smoke_%j.err

# EAAI2 远程冒烟测试：N=16 端到端验证 GPU 训练流程
set -e

source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2

mkdir -p logs results

echo "=== 环境 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== 冒烟测试: N=16, 16样本, 3 epoch, GPU ==="
python code/train.py \
    --nx 16 --ny 16 --nz 16 \
    --samples 16 --epochs 3 --batch 4 \
    --device cuda \
    --out results/checkpoint_smoke.pt

echo "=== 完成 ==="
ls -la results/
