#!/bin/bash
#SBATCH --job-name=EAAI2_synth
#SBATCH --partition=cu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/synth_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/synth_%j.err

# EAAI2 并行数据合成：N=16 网格，300 样本，16 进程并行
set -e

source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2

export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH

mkdir -p data logs

echo "=== 环境 ==="
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

echo "=== 并行合成 300 样本, 16 进程 ==="
python code/data_generation/synthesize_parallel.py \
    --nx 16 --ny 16 --nz 16 \
    --n_samples 300 --n_proc 16 \
    --max_iter 30 \
    --out data/train_16.npz

echo "=== 完成 ==="
ls -la data/
