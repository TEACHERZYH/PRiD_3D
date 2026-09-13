#!/bin/bash
#SBATCH --job-name=EAAI2_ms15
#SBATCH --partition=fat
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/multiseed_ext_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/multiseed_ext_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# 多 seed 扩展验证（15 种子）：确认
#   (a) PRiD 方差显著小于 no_bridge（初步 n=5: sd 4.9x, F p=0.0092）
#   (b) PRiD vs no_bridge 均值差是否达显著（初步 Δ+0.0170, p=0.128, d=0.86）
#   含前 5 个种子（与 job 34941 一致），以便合并/核对
python -u code/eval_multiseed_stability.py \
  --seeds 12000,22000,32000,42000,52000,62000,72000,82000,92000,102000,112000,122000,132000,142000,152000 \
  --n_samples 30 --refine_offset 65000 --K 5 --t_refine 30 --ddim_steps 20 \
  --degenerate_occ 10 \
  --out data/multiseed15_stability.csv --detail_out data/multiseed15_detail.csv
echo "15 种子扩展验证完成"
