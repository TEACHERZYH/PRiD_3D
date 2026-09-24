#!/bin/bash
#SBATCH --job-name=EAAI2_volnorm
#SBATCH --partition=fat
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/volnorm_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/volnorm_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 训练分布对照的**体积归一化**补做评估
#   动机：等材料量注入对齐了训练集材料量，但 dirty 模型生成占用仍低 46%，
#   而连通性天然依赖占用 → 原口径下的连通性差无法归因。
#   本评估按工况目标体积分数 V0 二值化（取密度最高的 k=V0·N 个体素），
#   使两条件的占用**按构造完全相等**。
for s in 1 2 3 4 5; do
  C="results/checkpoint_16_eq${s}_clean.pt"
  D="results/checkpoint_16_eq${s}_dirty.pt"
  if [ ! -f "${C}" ] || [ ! -f "${D}" ]; then
    echo "!! 缺少 seed ${s} 的 checkpoint，跳过"
    continue
  fi
  echo "=== volnorm eval seed ${s} ==="
  python -u code/eval_dist_volnorm.py \
    --data data/train_16.npz --n_samples 30 \
    --clean "${C}" --dirty "${D}" \
    --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
    --out "data/dist_volnorm_eq${s}.csv"
done
echo "体积归一化评估完成"
