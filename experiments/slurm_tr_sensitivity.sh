#!/bin/bash
#SBATCH --job-name=EAAI2_tr_sens
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/tr_sens_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/tr_sens_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# t_r 敏感性：低噪重构的"保真-修复"权衡（对接 SDEdit 的 t0 权衡）
#   固定 unrefined 场，仅变重构噪声 t_r（K=5，bridge=False）
python -u code/eval_tr_sensitivity.py \
  --n_samples 30 --t_r_list 10,20,30,50,80,120,160,199 \
  --K 5 --ddim_steps 20 --seed 12000 --refine_seed 77000 \
  --out data/tr_sensitivity.csv
echo "t_r 敏感性实验完成"
