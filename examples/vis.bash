cd /workspace/vggt
PYTHONPATH=/workspace/vggt:/workspace/vggt/training python3 -m training.data.datasets.coolant \
    --data_dir /workspace/Coolant_dataset/lidar-depth-poses \
    --num_frames 10 \
    --depth_stride 4 \
    --point_size 1.5