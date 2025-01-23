#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-1,worker-2,worker-3,worker-4,worker-5,worker-6,worker-7,worker-8
#SBATCH --output=/home/wiss/zhang/code/open-eqa/slurm/baseline_llama_4gpu-%j.out 
#SBATCH --partition all

source /home/wiss/zhang/anaconda3/bin/activate openeqa

date
hostname
which python

export NCCL_P2P_DISABLE=1

MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# python -m torch.distributed.run --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE:$MASTER_PORT \
#     /home/wiss/zhang/code/open-eqa/openeqa/baselines/llama.py \
#     --seed 4321 \
#     --model-path /nfs/data2/zhang/projects/openeqa/llama/Llama-2-7b-hf \

python /home/wiss/zhang/code/open-eqa/openeqa/baselines/llama_distributed.py \
    --seed 8123 \
    --model-path /nfs/data2/zhang/projects/openeqa/llama/Llama-2-70b-hf \
    --load-in-8bit \

