#!/bin/bash
#SBATCH --job-name=llava_hm3d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-1,worker-2,worker-3,worker-4,worker-8
#SBATCH --output=//home/wiss/zhang/code/open-eqa/Llava-1.5/slurm/output-%j.out
#SBATCH --partition major

# srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=12:00:00 --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-1,worker-2,worker-3,worker-4,worker-8 --pty bash

source /home/wiss/zhang/anaconda3/bin/activate iclblip

date
hostname
which python

export NCCL_P2P_DISABLE=1

MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo "Running on MASTER_NODE=$MASTER_NODE, MASTER_PORT=$MASTER_PORT, RDZV_ID=$RDZV_ID"

python -m torch.distributed.run --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE:$MASTER_PORT \
  /home/wiss/zhang/code/open-eqa/Llava-1.5/llava_caption.py \