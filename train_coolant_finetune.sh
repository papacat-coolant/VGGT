#!/bin/bash
# Finetune VGGT on Coolant single scene
# This script trains on random consecutive 10-frame sequences, rescaled to 518x518

cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=/workspace/vggt:/workspace/vggt/training:$PYTHONPATH

# Configuration
CONFIG="training/config/finetune_coolant.yaml"
PRETRAINED_CKPT="${1:-/path/to/pretrained/vggt/checkpoint.pth}"  # Optional: pass checkpoint path as first argument

# Create log directory
mkdir -p logs/finetune_coolant_single_scene

echo "=============================================="
echo "Finetuning VGGT on Coolant Single Scene"
echo "=============================================="
echo "Config: $CONFIG"
echo "Data: /workspace/Coolant_dataset/lidar-depth-poses"
echo "Image size: 518x518"
echo "Frames per batch: 10 consecutive frames"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "=============================================="
echo ""

# Run training
python3 training/train.py \
    --config-name finetune_coolant \
    checkpoint.resume_checkpoint_path="$PRETRAINED_CKPT" \
    "$@"

echo ""
echo "Training completed!"
echo "Logs saved to: logs/finetune_coolant_single_scene/"

