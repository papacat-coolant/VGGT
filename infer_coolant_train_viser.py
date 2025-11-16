#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for Coolant training dataset using trained checkpoint with nerfvis visualization.

Usage:
    python infer_coolant_train_viser.py \
        --checkpoint /workspace/vggt/logs/finetune_coolant_single_scene/ckpts/checkpoint.pt \
        --data_dir /workspace/Coolant_dataset/lidar-depth-poses \
        --sequence_name <sequence_name> \
        --num_frames 10 \
        --port 8080
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import nerfvis

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

# Import sky segmentation functions from demo_viser if needed
try:
    from demo_viser import apply_sky_segmentation
except ImportError:
    apply_sky_segmentation = None


def nerfvis_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 25.0,  # represents percentage (e.g., 25 means filter lowest 25%)
    use_point_map: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    frustum_depth: float = 2.0,
    point_size: float = 1.5,
    depth_stride: int = 4,
):
    """
    Visualize predicted 3D points and camera poses with nerfvis.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the nerfvis server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
        frustum_depth (float): Depth extent for camera frustums.
        point_size (float): Size of points in visualization.
        depth_stride (int): Stride for sampling depth points (higher = fewer points).
    """
    print(f"Starting nerfvis visualization on port {port}")
    
    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    
    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map
    
    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None and apply_sky_segmentation is not None:
        conf = apply_sky_segmentation(conf, image_folder)
    
    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape
    
    # Convert extrinsics from world-to-camera (w2c, 3x4) to camera-to-world (c2w, 4x4) for nerfvis
    # VGGT outputs w2c format, nerfvis expects c2w format
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
    
    # Create nerfvis scene
    scene = nerfvis.Scene("VGGT Coolant Visualization", default_opencv=True)
    scene.set_opencv()
    scene.set_opencv_world()
    
    # Flatten points and confidences for filtering
    points_flat = world_points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    
    # Compute confidence threshold
    if init_conf_threshold == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf_flat, init_conf_threshold)
    
    print(f"Confidence threshold (percentile {init_conf_threshold}%): {conf_threshold}")
    
    # Filter points by confidence
    conf_mask = (conf_flat >= conf_threshold) & (conf_flat > 1e-5)
    
    # Visualize each frame
    print("Adding cameras and point clouds to scene...")
    for idx in tqdm(range(S)):
        # Get camera-to-world matrix for this frame
        c2w = cam_to_world_mat[idx]  # (4, 4)
        
        # Extract rotation and translation
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        
        # Get image and intrinsics
        image = colors[idx]  # (H, W, 3) in [0, 1]
        rgb = (image * 255).astype(np.uint8)  # (H, W, 3) in [0, 255]
        intrinsic = intrinsics_cam[idx]
        fx = intrinsic[0, 0]
        
        frame_id = f"frame_{idx:05d}"
        
        # Add camera frustum
        scene.add_camera_frustum(
            f"camera/{frame_id}/frustum",
            r=R_c2w,
            t=t_c2w,
            focal_length=float(fx),
            image_width=W,
            image_height=H,
            z=float(frustum_depth) * 0.1,
        )
        
        # Add RGB billboard
        scene.add_image(
            f"camera/{frame_id}/image",
            rgb,
            r=R_c2w,
            t=t_c2w,
            focal_length=float(fx),
            z=float(frustum_depth) * 0.1,
            image_size=min(256, max(W, H)),  # Reduced to 256 for smaller billboard
        )
        
        # Add point cloud for this frame
        # Sample points with stride to reduce density
        sampled_points = world_points[idx, ::depth_stride, ::depth_stride]  # (H//stride, W//stride, 3)
        sampled_colors = rgb[::depth_stride, ::depth_stride]  # (H//stride, W//stride, 3)
        sampled_conf = conf[idx, ::depth_stride, ::depth_stride]  # (H//stride, W//stride)
        
        # Get valid points (confidence > threshold)
        valid_mask = (sampled_conf >= conf_threshold) & (sampled_conf > 1e-5)
        if np.any(valid_mask):
            pts_world = sampled_points[valid_mask]
            colors_valid = (sampled_colors[valid_mask] / 255.0)  # Normalize to [0, 1]
            
            # Add point cloud
            scene.add_points(
                f"points/{frame_id}",
                pts_world,
                point_size=point_size,
                vert_color=colors_valid,
            )
    
    # Add coordinate axes
    scene.add_axes()
    
    print("\n" + "="*60)
    print("Visualization ready! Opening viewer...")
    print(f"Port: {port}")
    print("="*60)
    
    # Display the scene
    scene.display(port=port)
    
    return scene


def load_checkpoint(checkpoint_path: str, model: VGGT, device: str):
    """
    Load model weights from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: VGGT model instance
        device: Device to load model on
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model state dict
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint
    
    # Load model weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
    
    model.eval()
    model = model.to(device)
    
    print("Checkpoint loaded successfully!")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    return model


def get_coolant_train_images(data_dir: str, sequence_name: str = None, num_frames: int = 10):
    """
    Get image paths from Coolant training dataset.
    
    Args:
        data_dir: Root directory of Coolant dataset
        sequence_name: Specific sequence name (if None, uses first available sequence)
        num_frames: Number of consecutive frames to load
    
    Returns:
        List of image file paths
    """
    # Find all sequence directories
    sequences = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            images_dir = os.path.join(item_path, "images")
            if os.path.isdir(images_dir):
                sequences.append((item, images_dir))
    
    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {data_dir}")
    
    # Select sequence
    if sequence_name is None:
        sequence_name, images_dir = sequences[0]
        print(f"No sequence specified, using first available: {sequence_name}")
    else:
        images_dir = None
        for seq_name, seq_images_dir in sequences:
            if seq_name == sequence_name:
                images_dir = seq_images_dir
                break
        if images_dir is None:
            raise ValueError(f"Sequence '{sequence_name}' not found. Available sequences: {[s[0] for s in sequences]}")
    
    print(f"Loading images from sequence: {sequence_name}")
    print(f"Images directory: {images_dir}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(image_files)} images in sequence")
    
    # Select consecutive frames
    num_frames = min(num_frames, len(image_files))
    start_index = 25
    stride = 2
    selected_images = image_files[start_index:num_frames*stride+start_index:stride]
    
    print(f"Selected {len(selected_images)} consecutive frames")
    
    return selected_images, sequence_name


def run_inference(model, image_paths, device):
    """
    Run inference on images.
    
    Args:
        model: VGGT model
        image_paths: List of image file paths
        device: Device to run inference on
    
    Returns:
        Dictionary containing predictions
    """
    print(f"Loading and preprocessing {len(image_paths)} images...")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    print("Processing model outputs...")
    # Convert tensors to numpy and remove batch dimension
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    
    # Remove pose_enc_list if present (not needed for visualization)
    if "pose_enc_list" in predictions:
        del predictions["pose_enc_list"]
    
    # Ensure depth has correct shape (S, H, W, 1)
    depth_map = predictions["depth"]  # (S, H, W, 1)
    if depth_map.ndim == 3:
        depth_map = depth_map[..., None]
    predictions["depth"] = depth_map
    
    # Generate world points from depth map
    print("Computing world points from depth map...")
    world_points = unproject_depth_map_to_point_map(
        depth_map, 
        predictions["extrinsic"], 
        predictions["intrinsic"]
    )
    predictions["world_points"] = world_points
    
    # Ensure depth_conf exists (use depth_conf from predictions or create from depth)
    if "depth_conf" not in predictions:
        # Create a simple confidence map (you might want to use actual confidence if available)
        print("Warning: depth_conf not found, using uniform confidence")
        depth_conf = np.ones_like(depth_map.squeeze(-1))  # (S, H, W)
        predictions["depth_conf"] = depth_conf
    
    # Ensure world_points_conf exists
    if "world_points_conf" not in predictions:
        predictions["world_points_conf"] = predictions["depth_conf"]
    
    # Store images for visualization - viser_wrapper expects (S, 3, H, W)
    if "images" not in predictions:
        # images is currently (B, S, 3, H, W) after squeeze(0) -> (S, 3, H, W)
        predictions["images"] = images.cpu().numpy().squeeze(0)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Coolant training dataset with nerfvis visualization"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., /workspace/vggt/logs/finetune_coolant_single_scene/ckpts/checkpoint.pt)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/Coolant_dataset/lidar-depth-poses",
        help="Root directory of Coolant dataset"
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        default=None,
        help="Specific sequence name (if None, uses first available sequence)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of consecutive frames to visualize"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number for nerfvis server"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=25.0,
        help="Initial percentage of low-confidence points to filter out"
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Use point map instead of depth-based points"
    )
    parser.add_argument(
        "--mask_sky",
        action="store_true",
        help="Apply sky segmentation to filter out sky points"
    )
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=2.0,
        help="Depth extent for camera frustums (smaller value = smaller frustum)"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help="Size of points in visualization"
    )
    parser.add_argument(
        "--depth_stride",
        type=int,
        default=4,
        help="Stride for sampling depth points (higher = fewer points)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU may be slow. Consider using CUDA if available.")
    
    # Initialize model
    print("Initializing VGGT model...")
    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
    )
    
    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model, device)
    
    # Get image paths from dataset
    image_paths, sequence_name = get_coolant_train_images(
        args.data_dir,
        args.sequence_name,
        args.num_frames
    )
    
    # Run inference
    predictions = run_inference(model, image_paths, device)
    
    # Prepare image folder path for sky segmentation if needed
    image_folder = os.path.dirname(image_paths[0]) if image_paths else None
    
    # Visualize with nerfvis
    print(f"Starting nerfvis visualization on port {args.port}...")
    print(f"Visualizing sequence: {sequence_name}")
    print(f"Number of frames: {len(image_paths)}")
    
    nerfvis_scene = nerfvis_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        mask_sky=args.mask_sky,
        image_folder=image_folder,
        frustum_depth=args.frustum_depth,
        point_size=args.point_size,
        depth_stride=args.depth_stride,
    )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Scene displayed on port {args.port}")
    print("="*60)


if __name__ == "__main__":
    main()

