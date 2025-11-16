#!/bin/bash
# Script to run inference on Coolant training dataset with viser visualization

cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=/workspace/vggt:/workspace/vggt/training:$PYTHONPATH

# Default values
CHECKPOINT="/workspace/vggt/logs/finetune_coolant_single_scene/ckpts/checkpoint.pt"
DATA_DIR="/workspace/Coolant_dataset/lidar-depth-poses"
NUM_FRAMES=10
PORT=8011
SEQUENCE_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --sequence_name)
            SEQUENCE_NAME="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--checkpoint PATH] [--data_dir PATH] [--sequence_name NAME] [--num_frames N] [--port PORT]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Running Inference on Coolant Training Dataset"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data directory: $DATA_DIR"
echo "Sequence name: ${SEQUENCE_NAME:-'auto (first available)'}"
echo "Number of frames: $NUM_FRAMES"
echo "Port: $PORT"
echo "=============================================="
echo ""

# Build command
CMD="python infer_coolant_train_viser.py \
    --checkpoint \"$CHECKPOINT\" \
    --data_dir \"$DATA_DIR\" \
    --num_frames $NUM_FRAMES \
    --port $PORT"

if [ -n "$SEQUENCE_NAME" ]; then
    CMD="$CMD --sequence_name \"$SEQUENCE_NAME\""
fi

# Run the command
eval $CMD

