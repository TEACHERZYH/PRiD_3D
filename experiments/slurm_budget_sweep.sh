#!/bin/bash
#SBATCH --job-name=EAAI2_budgetsweep
#SBATCH --partition=fat
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/budget_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 体积重标定细粒度扫描：逼近 SIMP 参考解（OOD 留出集）
SEEDS="${SEEDS:-12000,22000,32000,42000,52000}"
python -u code/explore_budget_refinement.py \
  --data data/ood_16.npz --ckpt results/checkpoint_16.pt \
  --n_samples 30 --seeds "${SEEDS}" --ddim_steps 20 \
  --out "data/ood_budget_sweep_${SEEDS//,/}.csv"
echo "体积重标定扫描完成（seeds=${SEEDS}）"
