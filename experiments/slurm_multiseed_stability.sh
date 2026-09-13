#!/bin/bash
#SBATCH --job-name=EAAI2_multiseed
#SBATCH --partition=fat
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/multiseed_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/multiseed_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# 多 seed 稳定性验证：确认「桥接 vs 重构」结论是否稳健（结论随噪声配对翻转，须重复）
#   5 个采样种子 × 30 样本 × 四格（unrefined/one-shot/no-bridge/PRiD）
#   报告种子级胜负、退化样本（no_bridge occ<=10）统计
python -u code/eval_multiseed_stability.py \
  --seeds 12000,22000,32000,42000,52000 --n_samples 30 \
  --refine_offset 65000 --K 5 --t_refine 30 --ddim_steps 20 \
  --degenerate_occ 10 \
  --out data/multiseed_stability.csv --detail_out data/multiseed_detail.csv
echo "多 seed 稳定性验证完成"
