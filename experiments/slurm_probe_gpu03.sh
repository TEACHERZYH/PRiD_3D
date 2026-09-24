#!/bin/bash
#SBATCH --job-name=EAAI2_probe
#SBATCH --partition=3090
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/probe_%j.log
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/logs/probe_%j.err
set -e
source /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/venv/bin/activate
cd /data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2
export PYTHONPATH=/data/home/xinxi-zhyh/xinxi-zhyh/projects/EAAI2/code:$PYTHONPATH
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

# 探针：逐步打印时间戳，定位 gpu03 上到底卡在哪一步
# （35304/35305 在 gpu03 上启动后 >7 分钟无任何输出，而 fat05 上同 venv 正常）
python -u - <<'PY'
import os, time
def t():
    return time.strftime('%H:%M:%S')
print(t(), 'start', flush=True)
print(t(), 'cwd', os.getcwd(), flush=True)
import numpy as np
print(t(), 'numpy', np.__version__, flush=True)
p = 'data/train_16.npz'
print(t(), 'npz size', os.path.getsize(p), flush=True)
d = np.load(p)
print(t(), 'npz loaded', d['conds'].shape, flush=True)
import torch
print(t(), 'torch', torch.__version__, flush=True)
x = torch.zeros(4, 4).cuda()
print(t(), 'cuda ok', float(x.sum()), torch.cuda.get_device_name(0), flush=True)
import scipy
from scipy.ndimage import label as cc
print(t(), 'scipy', scipy.__version__, 'cc ok', flush=True)
print(t(), 'probe finished', flush=True)
PY
echo "probe done"
