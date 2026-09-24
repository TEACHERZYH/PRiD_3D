#!/bin/bash
#SBATCH --job-name=EAAI2_ood_ctl
#SBATCH --partition=fat
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood_ctl_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood_ctl_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 在留出 OOD 集上重跑「负对照」与训练分布对照（表 6、§5.7）
# 参数与原协议逐字相同，只替换 --data 与 --out。
D=data/ood_16.npz

echo "=== 6. 软约束对照（3 seed）==="
for s in 12000 22000 32000; do
  python -u code/soft_constraint_baseline.py --data ${D} \
    --ckpt results/checkpoint_16.pt --n_samples 30 --seed "${s}" \
    --guide_steps 120 --guide_lr 0.7 --guide_iters 60 --vol_weight 0.5 \
    --out "data/ood_soft_vs_proj_seed${s}.csv"
done

echo "=== 7. in-loop vs post-hoc（3 seed）==="
python -u code/eval_inloop_vs_posthoc.py --data ${D} \
  --ckpt results/checkpoint_16.pt --n_samples 30 --seeds 12000,22000,32000 \
  --ddim_steps 20 --out data/ood_inloop_vs_posthoc.csv

echo "=== 8. 训练分布对照（5 seed，0.5 阈值口径）==="
for s in 1 2 3 4 5; do
  C="results/checkpoint_16_eq${s}_clean.pt"; Dy="results/checkpoint_16_eq${s}_dirty.pt"
  [ -f "${C}" ] && [ -f "${Dy}" ] || { echo "跳过 seed ${s}（缺 checkpoint）"; continue; }
  python -u code/eval_clean_vs_dirty.py --data ${D} --n_samples 30 \
    --clean "${C}" --dirty "${Dy}" \
    --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
    --out "data/ood_clean_vs_dirty_eq${s}.csv"
done

echo "=== 9. 训练分布对照（体积归一化）==="
for s in 1 2 3 4 5; do
  C="results/checkpoint_16_eq${s}_clean.pt"; Dy="results/checkpoint_16_eq${s}_dirty.pt"
  [ -f "${C}" ] && [ -f "${Dy}" ] || continue
  python -u code/eval_dist_volnorm.py --data ${D} --n_samples 30 \
    --clean "${C}" --dirty "${Dy}" \
    --K 5 --t_refine 30 --seed 12000 --refine_seed 77000 \
    --out "data/ood_dist_volnorm_eq${s}.csv"
done

echo "OOD 对照评测完成"
