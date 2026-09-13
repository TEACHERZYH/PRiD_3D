#!/bin/bash
#SBATCH --job-name=EAAI2_clndirty
#SBATCH --partition=3090
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/clean_dirty_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/clean_dirty_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 步骤 1：对照训练（clean 标签 vs 注入浮材的 dirty 标签，同划分同配置）
#    epochs=120 与现有 checkpoint_16.pt 的训练配置一致（slurm_train_eval.sh）
python -u code/train_clean_vs_dirty.py \
  --data data/train_16.npz --epochs 120 --batch 8 --lr 1e-3 \
  --float_ratio 0.15 --block 2 --seed 20260912 --out_dir results/
echo "对照训练完成"

# 步骤 2：对照评估（同一批条件与采样种子）
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_16_clean.pt --dirty results/checkpoint_16_dirty.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/clean_vs_dirty.csv
echo "对照评估完成"
