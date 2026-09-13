#!/bin/bash
#SBATCH --job-name=EAAI2_feas_attr
#SBATCH --partition=fat
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/feas_attr_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/feas_attr_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# P2 可行性吸引子：破坏可行解 -> 加噪到 t0 -> 去噪 -> 度量恢复
#   可行起点 = 模型自身重构输出（bridge=False, K=5）
#   破坏 = 三轴搜索使连通性最低的单层切片并删除
#   t0 扫描 = 保真-恢复权衡曲线
python -u code/eval_feasibility_projection.py --mode attractor \
  --n_samples 30 --t0_list 20,40,60,90,120,160 \
  --seed 12000 --refine_seed 77000 --rec_seed 31000 \
  --K 5 --t_refine 30 --ddim_steps 20 \
  --out data/feasibility_attractor.csv
echo "可行性吸引子实验完成"

# P1 直接证据：自然采样轨迹上的可行性演化
python -u code/eval_feasibility_projection.py --mode trajectory \
  --n_samples 20 --seed 12000 --ddim_steps 20 \
  --out data/trajectory_feasibility.csv
echo "去噪轨迹实验完成"
