#!/bin/bash
#SBATCH --job-name=Llama-2-70b-hf-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/open-eqa/slurm/matrix_llama-70b-%j.out
#SBATCH --partition a40

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate openeqa

# srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:a100:4 -C a100_80 --time=4:00:00 --partition=a100 --pty bash

date
hostname
which python


export NCCL_P2P_DISABLE=1
export WANDB_DISABLED=1


MASTER_ADDR=localhost
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
python -m torch.distributed.run --nproc_per_node=4 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    /home/hpc/v100dd/v100dd12/code/open-eqa/evaluate-predictions.py \
    /home/wiss/zhang/code/open-eqa/data/results/open-eqa-v0-llama-2-7b-hf-4321.json \
    --output-directory /home/hpc/v100dd/v100dd12/code/open-eqa/data/matrics \

# 停止监控
kill $GPU_MONITOR_PID $CPU_MONITOR_PID
