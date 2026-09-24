#!/bin/bash
#SBATCH --job-name=EAAI2_synth32
#SBATCH --partition=fat
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/synth32_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/synth32_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 32³ 训练集合成（SIMP 拓扑优化，300 例，4 通道条件：V0/应力/荷载/支座）
# 分辨率扩展：16³ → 32³（8× 体素），max_iter 提高到 60 以保证 32³ 下收敛
python -u code/data_generation/synthesize_parallel.py \
  --nx 32 --ny 32 --nz 32 --n_samples 300 --n_proc 16 \
  --volfrac_min 0.2 --volfrac_max 0.5 --max_iter 60 \
  --out data/train_32.npz
echo "32³ 训练集合成完成"
ls -la data/train_32.npz
