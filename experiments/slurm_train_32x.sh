#!/bin/bash
#SBATCH --job-name=EAAI2_t32x
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train32x_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/train32x_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# 32³ 重训：对抗后验坍缩（默认 4000 步太少，模型坍缩到稀疏均值 sample_mean~0.12）
# 三个旋钮通过环境变量注入：EPOCHS / LR / BASE / DEPTH / BATCH / OUT
EPOCHS="${EPOCHS:-600}"
LR="${LR:-3e-4}"
BASE="${BASE:-32}"
DEPTH="${DEPTH:-2}"
BATCH="${BATCH:-8}"
OUT="${OUT:-results/checkpoint_32x.pt}"

python -u code/train.py \
  --data data/train_32.npz --epochs "${EPOCHS}" --lr "${LR}" --batch "${BATCH}" \
  --device cuda --timesteps 200 --base "${BASE}" --depth "${DEPTH}" \
  --out "${OUT}"
echo "32³ 重训完成: ${OUT}"
