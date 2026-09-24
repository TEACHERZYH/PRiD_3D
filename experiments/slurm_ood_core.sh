#!/bin/bash
#SBATCH --job-name=EAAI2_ood_core
#SBATCH --partition=fat
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood_core_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/ood_core_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# 在**真正留出的 OOD 集**上重跑核心评测（ood_16.npz：30 个工况，与训练集零重合，
# 三类分布偏移各 10 例：volfrac / domain / load；标签 30/30 全连通）
# 原则：参数与论文原协议**逐字相同**，只替换 --data 与 --out。
D=data/ood_16.npz

echo "=== 1. 传递链 + DDIM 扫描（表 2、5.5 采样预算）==="
python -u code/eval_bridge_recon_factorial.py --data ${D} \
  --n_samples 30 --seed 12000 --refine_seed 77000 \
  --K 5 --t_refine 30 --ddim_steps 20 --ddim_scan 20,50,100 \
  --out data/ood_bridge_recon_factorial.csv \
  --scan_out data/ood_ddim_strength_scan.csv

echo "=== 2. 15 seed 因子/稳定性（表 4、表 5）==="
python -u code/eval_multiseed_stability.py --data ${D} \
  --seeds 12000,22000,32000,42000,52000,62000,72000,82000,92000,102000,112000,122000,132000,142000,152000 \
  --n_samples 30 --refine_offset 65000 --K 5 --t_refine 30 --ddim_steps 20 \
  --degenerate_occ 10 \
  --out data/ood_multiseed15_stability.csv --detail_out data/ood_multiseed15_detail.csv

echo "=== 3. t_r 扫描（表 7）==="
python -u code/eval_tr_sensitivity.py --data ${D} \
  --n_samples 30 --t_r_list 10,20,30,50,80,120,160,199 \
  --K 5 --ddim_steps 20 --seed 12000 --refine_seed 77000 \
  --out data/ood_tr_sensitivity.csv

echo "=== 4. 可行性吸引子（表 3）==="
python -u code/eval_feasibility_projection.py --mode attractor --data ${D} \
  --n_samples 30 --t0_list 20,40,60,90,120,160 \
  --seed 12000 --refine_seed 77000 --rec_seed 31000 \
  --K 5 --t_refine 30 --ddim_steps 20 \
  --out data/ood_feasibility_attractor.csv

echo "=== 5. 去噪轨迹（可行性窗口）==="
python -u code/eval_feasibility_projection.py --mode trajectory --data ${D} \
  --n_samples 20 --seed 12000 --ddim_steps 20 \
  --out data/ood_trajectory_feasibility.csv

echo "OOD 核心评测完成"
