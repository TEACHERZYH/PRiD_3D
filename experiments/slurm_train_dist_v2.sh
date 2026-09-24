#!/bin/bash
#SBATCH --job-name=EAAI2_dist2
#SBATCH --partition=3090
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/dist2_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/dist2_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 增补实验 3：训练分布对照 v2 —— 等材料量注入 + 多 seed
#   修复的缺陷（论文 §5.7）：
#     ① 「训练分布决定恢复上限」的因果实验 n=1/条件，功效不足（p=0.175 不可判）；
#     ② 原 dirty 条件把训练集材料量抬高 15%，与「材料多→生成多→更易连通」混淆；
#     ③ 原脚本注释称「相同划分」，但划分未播种，clean/dirty 实际划分不同。
#   本版：--match_volume 等材料量（注入浮材后按最低 von Mises 应力从主体删等量）、
#         划分与初始化由 seed 唯一决定、每条件 5 个 seed。
SEEDS="${SEEDS:-1 2 3 4 5}"
for s in ${SEEDS}; do
  echo "=== seed 2026090${s} ==="
  python -u code/train_clean_vs_dirty.py \
    --data data/train_16.npz --epochs 120 --batch 8 --lr 1e-3 \
    --float_ratio 0.15 --block 2 --seed "2026090${s}" \
    --match_volume --tag "eq${s}" --out_dir results/ --device cuda
done
echo "等材料量训练分布对照（训练部分）完成"
