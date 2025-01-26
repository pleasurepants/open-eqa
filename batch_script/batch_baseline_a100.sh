#!/bin/bash
#SBATCH --job-name=Llama-4a100-80
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4 -C a100_80
#SBATCH --ntasks=4  
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/open-eqa/slurm/baseline_llama_4a100_80-%j.out
#SBATCH --partition=a100

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate openeqa

# srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:a100:4 -C a100_80 --time=4:00:00 --partition=a100 --pty bash
# srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:a100:8 --time=4:00:00 --partition=a100 --pty bash
date
hostname
which python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export WANDB_DISABLED=1
export OMP_NUM_THREADS=16  

MASTER_ADDR=localhost
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
# python -m torch.distributed.run --nproc_per_node=4 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#     /home/hpc/v100dd/v100dd12/code/open-eqa/openeqa/baselines/llama_distributed.py \
#     --seed 4321 \
#     --model-path /anvme/workspace/v100dd12-openeqa/llama/Llama-2-70b-hf \
#     --load-in-8bit \
python /home/hpc/v100dd/v100dd12/code/open-eqa/openeqa/baselines/llama_distributed.py \
    --seed 4480 \
    --model-path /anvme/workspace/v100dd12-openeqa/llama/Llama-2-70b-hf \
    --load-in-8bit \
