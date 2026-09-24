#!/bin/bash
#SBATCH --job-name=EAAI2_explore
#SBATCH --partition=fat
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/explore_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/explore_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 改进方案探索：一组精炼变体作用在同一采样输出上（严格配对）
#   ⚠️ sbatch --export 以**逗号**分隔多个变量赋值，值里出现逗号会被截断！
#      故本脚本对外用 **+ / 空格** 作为值内分隔符，脚本内再转成逗号。
#      （已两次踩坑：SEEDS=12000,22000,32000 只传了 12000；VARIANT_LIST 同理）
SEEDS="${SEEDS:-12000 22000 32000}"
VARIANT_LIST="${VARIANT_LIST:-}"
SEEDS_CSV="${SEEDS// /,}"
TAG="${SEEDS// /_}"
# 变体列表也进文件名，避免同名覆盖（2026-09-17 踩坑）
if [ -n "${VARIANT_LIST}" ]; then TAG="${TAG}_${VARIANT_LIST//+/_}"; fi
if [ -n "${VARIANT_LIST}" ]; then
  TMP="${VARIANT_LIST//+/ }"          # 允许用 + 分隔
  VARG="--variants ${TMP// /,}"
else
  VARG=""
fi

python -u code/explore_refine_variants.py \
  --data ${DATA:-data/train_16.npz} --ckpt results/checkpoint_16.pt \
  --n_samples 30 --seeds "${SEEDS_CSV}" --ddim_steps 20 \
  ${VARG} \
  --out "data/${PREFIX:-explore_variants}_${TAG}.csv"
echo "改进方案探索完成（seeds=${SEEDS}，variants=${VARIANT_LIST:-all}）"
