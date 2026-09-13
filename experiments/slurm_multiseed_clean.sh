#!/bin/bash
#SBATCH --job-name=EAAI2_msclean
#SBATCH --partition=3090
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/msclean_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/msclean_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 多 seed 重训诊断：判断"新训 clean 模型采样连通性 0.7533"是运气差还是训练高方差
python -u code/train_multiseed_clean.py \
  --seeds 1,2,3,4,5 --epochs 120 --batch 8 --lr 1e-3 \
  --out_dir results/ms_clean
echo "多 seed clean 训练诊断完成"
