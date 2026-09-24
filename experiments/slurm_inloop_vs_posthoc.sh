#!/bin/bash
#SBATCH --job-name=EAAI2_inloop
#SBATCH --partition=fat
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/inloop_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/inloop_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 增补实验 2：采样中投影（in-loop）vs 采样后投影（post-hoc）配对对照
#   修复的缺陷：论文 §5.5「in-loop 仅 13.3% 配对占优」与结论「87% 更差」
#   原为旧 P0 实验的单 seed 记录（experiments/实验记录（补做P0）.md），本机不可重算。
#   此处 3 seed × 30 样本 × 3 条件（post-hoc / in-loop@0.9 / in-loop@0.5），
#   严格配对初始噪声，产出逐样本 CSV。
python -u code/eval_inloop_vs_posthoc.py \
  --data data/train_16.npz --ckpt results/checkpoint_16.pt \
  --n_samples 30 --seeds 12000,22000,32000 --ddim_steps 20 \
  --out data/inloop_vs_posthoc.csv
echo "in-loop vs post-hoc 对照完成"
