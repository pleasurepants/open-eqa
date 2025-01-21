#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-2,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-1,worker-2,worker-3,worker-4,worker-5
#SBATCH --output=/home/wiss/zhang/code/open-eqa/slurm/baseline_llama-%j.out 
#SBATCH --partition major

source /home/wiss/zhang/anaconda3/bin/activate openeqa

date
hostname
which python

export NCCL_P2P_DISABLE=1

MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo "Running on MASTER_NODE=$MASTER_NODE, MASTER_PORT=$MASTER_PORT, RDZV_ID=$RDZV_ID"

# # 添加 GPU 和 CPU 使用情况监控
# echo "Monitoring CUDA and CPU usage..."
# nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 5 &
# GPU_MONITOR_PID=$!

# while true; do
#     top -b -n1 | grep -E "Cpu|%Cpu|Mem|%Mem" | head -n5
#     sleep 10
# done & 
# CPU_MONITOR_PID=$!

# 运行主程序
python -m torch.distributed.run --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE:$MASTER_PORT \
    /home/wiss/zhang/code/open-eqa/openeqa/baselines/llama.py \
    --seed 4321 \
    --model-path /nfs/data2/zhang/projects/openeqa/llama/Llama-2-7b-hf \

# 停止监控
kill $GPU_MONITOR_PID $CPU_MONITOR_PID
