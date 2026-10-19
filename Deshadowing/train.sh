#!/usr/bin/env bash

# Default config for Deshadowing
CONFIG=${1:-"Deshadowing/Options/Deshadowing_Restormer.yml"}
NUM_GPUS=${2:-1}
PORT=${3:-4321}

echo "Starting Restormer Deshadowing Training..."
echo "Config: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -le 1 ]; then
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt "$CONFIG"
else
    # Multi-GPU training
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUS" --master_port="$PORT" basicsr/train.py -opt "$CONFIG" --launcher pytorch
fi
