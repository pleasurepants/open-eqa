#!/bin/bash
#SBATCH --job-name=Llama-2-70b-hf
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/open-eqa/slurm/baseline_llama-%j.out
#SBATCH --partition a100

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate openeqa

# srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:a100:4 --time=4:00:00 --partition=a100 --pty bash

date
hostname
which python


export NCCL_P2P_DISABLE=1
export WANDB_DISABLED=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=10
export PYDEVD_WARN_EVALUATION_TIMEOUT=10

MASTER_ADDR=localhost
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
python -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
 -m torch.distributed.run --nproc_per_node=4 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    /home/hpc/v100dd/v100dd12/code/open-eqa/openeqa/baselines/llama.py \
    --seed 4321 \
    --model-path /anvme/workspace/v100dd12-openeqa/llama/Llama-2-70b-hf \

