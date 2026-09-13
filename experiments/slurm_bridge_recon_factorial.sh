#!/bin/bash
#SBATCH --job-name=EAAI2_bridge_recon
#SBATCH --partition=fat
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/bridge_recon_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/bridge_recon_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# 2x2 因子对照：分离「桥接投影」与「低噪重构」的贡献
#   四格：unrefined / one-shot(只桥接) / no-bridge(只重构) / PRiD(桥接+重构)
#   严格配对：同一初始场 + 相同 refine 噪声种子
# 附带：unrefined 采样强度扫描 ddim_steps in {20,50,100}
python -u code/eval_bridge_recon_factorial.py \
  --n_samples 30 --seed 12000 --refine_seed 77000 \
  --K 5 --t_refine 30 --ddim_steps 20 --ddim_scan 20,50,100 \
  --out data/bridge_recon_factorial.csv \
  --scan_out data/ddim_strength_scan.csv
echo "2x2 因子对照完成"
