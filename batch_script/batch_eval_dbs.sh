#!/bin/bash
#SBATCH --job-name=llama-70b-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-1,worker-2,worker-3,worker-4,worker-5,worker-6,worker-7,worker-8
#SBATCH --output=/home/wiss/zhang/code/open-eqa/slurm/matrix-70b-%j.out 
#SBATCH --partition all

source /home/wiss/zhang/anaconda3/bin/activate openeqa

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
python /home/wiss/zhang/code/open-eqa/evaluate-predictions.py \
    /home/wiss/zhang/code/open-eqa/data/results/open-eqa-v0-llama-2-70b-hf-4480.json \
    --output-directory /home/wiss/zhang/code/open-eqa/data/matrix \
