#!/bin/bash
#SBATCH --job-name=EAAI2_no_bridge
#SBATCH --partition=fat
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/no_bridge_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/no_bridge_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# no-bridge + reconstruction 对照：验证桥接（连通性投影）的必要性
# 对照 unrefined / no-bridge(纯重构) / PRiD(桥接+重构)，配对 seed=12000
python -u code/eval_no_bridge_control.py --n_samples 30 --seed 12000 --K 5 --t_refine 30 --out data/no_bridge_control.csv
echo "no-bridge 对照完成"
