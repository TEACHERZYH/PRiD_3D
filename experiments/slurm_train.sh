#!/bin/bash
#SBATCH --job-name=EAAI2_train
#SBATCH --partition=3090
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train_%j.err

# EAAI2 正式训练：流形保持扩散，N=16 网格
set -e

source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2

export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH

echo "=== 环境 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== 训练 ==="
python code/train.py \
    --data data/train_16.npz \
    --epochs 50 \
    --batch 8 \
    --device cuda \
    --timesteps 200 \
    --base 32 \
    --depth 2 \
    --out results/checkpoint_16.pt

echo "=== 完成 ==="
ls -la results/
