#!/bin/bash
#SBATCH --job-name=EAAI2_cpurepro
#SBATCH --partition=fat
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cpurepro_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cpurepro_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 复刻原训练配置（原 checkpoint_16.pt = train.py --epochs 120 --device cpu）
echo "=== 复刻训练（CPU，原配置）==="
python -u code/train.py \
  --data data/train_16.npz --epochs 120 --batch 8 --device cpu \
  --timesteps 200 --base 32 --depth 2 \
  --out results/checkpoint_repro_cpu.pt
echo "CPU 复刻训练完成"

# 等待文件系统落盘
sleep 15

echo "=== 诊断：复刻模型 vs 原模型 ==="
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_repro_cpu.pt --dirty results/checkpoint_16.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/repro_cpu_vs_orig.csv
echo "复刻诊断完成"
