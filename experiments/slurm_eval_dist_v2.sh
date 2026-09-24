#!/bin/bash
#SBATCH --job-name=EAAI2_dist2ev
#SBATCH --partition=fat
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/dist2ev_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/dist2ev_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 增补实验 3 的评估部分：对 5 对（clean, dirty-equal-volume）checkpoint 逐一评估
#   配对条件：同 seed 的 clean/dirty 使用同一划分与同一初始化，只差标签；
#   采样种子与噪声在两者间严格配对（脚本内 torch.manual_seed(seed+i)）。
SEEDS="${SEEDS:-1 2 3 4 5}"
for s in ${SEEDS}; do
  C="results/checkpoint_16_eq${s}_clean.pt"
  D="results/checkpoint_16_eq${s}_dirty.pt"
  if [ ! -f "${C}" ] || [ ! -f "${D}" ]; then
    echo "!! 缺少 seed ${s} 的 checkpoint，跳过（${C} / ${D}）"
    continue
  fi
  echo "=== eval seed ${s} ==="
  python -u code/eval_clean_vs_dirty.py \
    --data data/train_16.npz --n_samples 30 \
    --clean "${C}" --dirty "${D}" \
    --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
    --out "data/clean_vs_dirty_eq${s}.csv"
done
echo "等材料量训练分布对照（评估部分）完成"
