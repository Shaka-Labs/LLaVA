#!/bin/bash
#SBATCH --gres=gpumem:80g
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=3:00:00

export CONSUL_HTTP_ADDR=""
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE="/cluster/scratch/ismails/huggingface"

/cluster/home/ismails/robotics/LLaVA/scripts/v1_5/finetune_task_lora.sh
