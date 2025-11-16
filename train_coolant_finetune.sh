#!/bin/bash
# Finetune VGGT on Coolant single scene
# This script trains on random consecutive 10-frame sequences, rescaled to 518x518

cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=/workspace/vggt:/workspace/vggt/training:$PYTHONPATH

# Create log directory
mkdir -p logs/finetune_coolant_single_scene

echo "=============================================="
echo "Finetuning VGGT on Coolant Single Scene"
echo "=============================================="
echo "Config: training/config/finetune_coolant.yaml"
echo "Data: /workspace/Coolant_dataset/lidar-depth-poses"
echo "Image size: 518x518"
echo "Frames per batch: 10 consecutive frames"
echo "=============================================="
echo ""

# Run training (checkpoint path is set in config file)
# To resume training:
# 1. Automatic resume: Set resume_checkpoint_path to null in config, 
#    and it will auto-resume from logs/finetune_coolant_single_scene/ckpts/checkpoint.pt
# 2. Manual resume: Set resume_checkpoint_path to your checkpoint file path in config
torchrun --nproc_per_node=1 training/launch.py --config finetune_coolant

