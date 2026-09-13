#!/bin/bash
#SBATCH --job-name=EAAI2_cd_cpu
#SBATCH --partition=fat
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cd_cpu_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/cd_cpu_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# clean vs dirty 对照（CPU，与原 checkpoint_16.pt 同设备同配置）
#   同批次训练 clean + dirty 各一个，唯一差异是训练标签的连通性
echo "=== 步骤1：对照训练（CPU，120 epochs）==="
python -u code/train_clean_vs_dirty.py \
  --data data/train_16.npz --epochs 120 --batch 8 --lr 1e-3 \
  --float_ratio 0.15 --block 2 --seed 20260912 --device cpu --out_dir results/
echo "对照训练完成"

sleep 15

echo "=== 步骤2：对照评估（含浮材率）==="
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_16_clean_cpu.pt --dirty results/checkpoint_16_dirty_cpu.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/clean_vs_dirty_cpu.csv
echo "对照评估完成"

sleep 15

echo "=== 步骤3：与原始模型三方对照（原模型作为第三个 tag）==="
python -u code/eval_clean_vs_dirty.py \
  --data data/train_16.npz --n_samples 30 \
  --clean results/checkpoint_16.pt --dirty results/checkpoint_16_dirty_cpu.pt \
  --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
  --out data/orig_vs_dirtycpu.csv
echo "三方对照完成"
