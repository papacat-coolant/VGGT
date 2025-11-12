# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset


class CoolantDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        COOLANT_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the CoolantDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            COOLANT_DIR (str): Directory path to Coolant data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If COOLANT_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if COOLANT_DIR is None:
            raise ValueError("COOLANT_DIR must be specified.")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = []  # set any invalid sequence names here

        self.data_store = {}
        self.min_num_images = min_num_images

        logging.info(f"COOLANT_DIR is {COOLANT_DIR}")

        self.COOLANT_DIR = COOLANT_DIR

        total_frame_num = 0

        # Scan for all sequence directories in COOLANT_DIR
        # Each sequence should have: images/, depths/, cam_from_worlds.npy, intrinsics.npy
        # Skip known non-data directories
        skip_dirs = {'camera', 'scripts', 'clouds', 'trajectories', 'run.sh'}
        
        for seq_name in os.listdir(COOLANT_DIR):
            seq_path = osp.join(COOLANT_DIR, seq_name)
            
            if not osp.isdir(seq_path):
                continue
            
            # Skip known non-data directories
            if seq_name in skip_dirs:
                continue

            # Check if this directory has required files
            cam_from_worlds_path = osp.join(seq_path, "cam_from_worlds.npy")
            intrinsics_path = osp.join(seq_path, "intrinsics.npy")
            images_dir = osp.join(seq_path, "images")
            depths_dir = osp.join(seq_path, "depths")

            if not all([
                osp.exists(cam_from_worlds_path),
                osp.exists(intrinsics_path),
                osp.isdir(images_dir),
                osp.isdir(depths_dir),
            ]):
                logging.debug(f"Skipping {seq_name}: missing required files")
                continue

            if seq_name in self.invalid_sequence:
                continue

            # Load camera parameters
            try:
                cam_from_worlds = np.load(cam_from_worlds_path)  # (N, 3, 4)
                intrinsics = np.load(intrinsics_path)  # (N, 3, 3)
                
                num_frames = cam_from_worlds.shape[0]
                
                if num_frames < min_num_images:
                    logging.warning(f"Skipping {seq_name}: only {num_frames} images (min required: {min_num_images})")
                    continue

                # Load image names if available
                image_names_path = osp.join(seq_path, "image_names.json")
                if osp.exists(image_names_path):
                    with open(image_names_path, 'r') as f:
                        image_names = json.load(f)
                else:
                    # Fallback: assume images are named 00000.jpg, 00001.jpg, etc.
                    image_names = [f"{i:05d}.jpg" for i in range(num_frames)]

                # Store sequence metadata
                self.data_store[seq_name] = {
                    "seq_path": seq_path,
                    "cam_from_worlds": cam_from_worlds,
                    "intrinsics": intrinsics,
                    "image_names": image_names,
                    "num_frames": num_frames,
                }

                total_frame_num += num_frames
                logging.info(f"Loaded sequence: {seq_name} with {num_frames} frames")

            except Exception as e:
                logging.error(f"Error loading sequence {seq_name}: {e}")
                continue

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Coolant Data size: {self.sequence_list_len} sequences")
        logging.info(f"{status}: Coolant Data total frames: {total_frame_num}")
        logging.info(f"{status}: Coolant Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]
        seq_path = metadata["seq_path"]
        num_frames = metadata["num_frames"]
        cam_from_worlds = metadata["cam_from_worlds"]
        intrinsics_all = metadata["intrinsics"]
        image_names = metadata["image_names"]

        if ids is None:
            ids = np.random.choice(
                num_frames, img_per_seq, replace=self.allow_duplicate_img
            )

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for frame_id in ids:
            # Load image
            image_name = image_names[frame_id]
            image_path = osp.join(seq_path, "images", image_name)
            image = read_image_cv2(image_path)

            # Load depth if required
            if self.load_depth:
                # Depth files are typically named with the same index as images
                depth_name = f"{frame_id:05d}.npy"
                depth_path = osp.join(seq_path, "depths", depth_name)
                
                if osp.exists(depth_path):
                    depth_map = np.load(depth_path)
                    # Ensure depth map has the right type
                    depth_map = depth_map.astype(np.float32)
                    
                    # Apply depth thresholding if needed
                    depth_map = threshold_depth_map(
                        depth_map, min_percentile=-1, max_percentile=98
                    )
                else:
                    logging.warning(f"Depth file not found: {depth_path}")
                    depth_map = None
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            
            # Get camera parameters for this frame
            # cam_from_worlds is (3, 4) - world to camera transformation
            cam_from_world = cam_from_worlds[frame_id]  # (3, 4)
            intri_opencv = intrinsics_all[frame_id]  # (3, 3)
            
            # Convert cam_from_world (3, 4) to full extrinsic matrix (4, 4)
            # cam_from_world transforms world coords to camera coords
            extri_opencv = np.eye(4, dtype=np.float32)
            extri_opencv[:3, :] = cam_from_world

            intri_opencv = intri_opencv.astype(np.float32)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "coolant"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch


if __name__ == "__main__":
    """
    Visualize Coolant dataset using nerfvis.
    
    Usage:
        cd /workspace/vggt
        python /workspace/vggt/training/data/datasets/coolant.py --data_dir /workspace/Coolant_dataset/lidar-depth-poses/Dataset_converted_test --num_frames 5
    """
    import argparse
    import sys
    from types import SimpleNamespace
    import nerfvis
    
    parser = argparse.ArgumentParser(description="Visualize Coolant dataset with nerfvis")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/Coolant_dataset/lidar-depth-poses",
        help="Path to Coolant dataset directory"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of consecutive frames to visualize"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Starting frame index"
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
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=20.0,
        help="Depth extent for camera frustums"
    )
    args = parser.parse_args()
    
    # Create configuration
    common_conf = SimpleNamespace(
        img_size=518,
        patch_size=14,
        augs=SimpleNamespace(scales=None),  # No augmentation for visualization
        rescale=True,  # Downsample images and depth to img_size=518
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        get_nearby=False,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=False
    )
    
    # Initialize dataset
    print(f"Loading Coolant dataset from: {args.data_dir}")
    dataset = CoolantDataset(
        common_conf=common_conf,
        split='test',
        COOLANT_DIR=args.data_dir,
        min_num_images=1,  # Allow any sequence length
        len_test=1000
    )
    
    if dataset.sequence_list_len == 0:
        print("Error: No sequences found in dataset!")
        sys.exit(1)
    
    # Get first sequence
    seq_name = dataset.sequence_list[0]
    # Load data - must be consecutive frames
    total_frames = dataset.data_store[seq_name]["num_frames"]
    start_frame = min(args.start_frame, total_frames - 1)
    end_frame = min(start_frame + args.num_frames, total_frames)
    num_frames = end_frame - start_frame
    # Get consecutive frame IDs
    frame_ids = np.arange(start_frame, end_frame)
    
    batch = dataset.get_data(
        seq_index=0,
        img_per_seq=len(frame_ids),
        ids=frame_ids,
        aspect_ratio=1.0
    )
    
    # Create nerfvis scene
    scene = nerfvis.Scene("Coolant Dataset Viewer", default_opencv=True)
    scene.set_opencv()
    scene.set_opencv_world()
    
    # Visualize each frame
    for idx in range(batch["frame_num"]):
        image = batch["images"][idx]  # (H, W, 3) in [0, 255]
        extrinsic = batch["extrinsics"][idx]  # (4, 4) world-to-camera
        intrinsic = batch["intrinsics"][idx]  # (3, 3)
        world_points = batch["world_points"][idx]  # (H, W, 3)
        point_mask = batch["point_masks"][idx]  # (H, W)
        
        # Convert extrinsic from world-to-camera to camera-to-world
        w2c = extrinsic
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -R_w2c.T @ t_w2c
        
        H, W = image.shape[:2]
        fx = intrinsic[0, 0]
        frame_id = f"frame_{frame_ids[idx]:05d}"
        
        # Add camera frustum
        scene.add_camera_frustum(
            f"camera/{frame_id}/frustum",
            r=c2w[:3, :3],
            t=c2w[:3, 3],
            focal_length=float(fx),
            image_width=W,
            image_height=H,
            z=float(args.frustum_depth),
        )
        
        # Add RGB billboard
        rgb = image.astype(np.uint8)
        scene.add_image(
            f"camera/{frame_id}/image",
            rgb,
            r=c2w[:3, :3],
            t=c2w[:3, 3],
            focal_length=float(fx),
            z=float(args.frustum_depth),
            image_size=min(1024, max(W, H)),
        )
        
        # Use pre-computed point cloud from batch
        if np.any(point_mask):
            # Sample points with stride to reduce density
            stride = args.depth_stride
            sampled_mask = point_mask[::stride, ::stride]
            sampled_points = world_points[::stride, ::stride]
            sampled_colors = rgb[::stride, ::stride]
            
            # Get valid points
            valid_idx = sampled_mask.reshape(-1)
            pts_world = sampled_points.reshape(-1, 3)[valid_idx]
            colors = (sampled_colors.reshape(-1, 3)[valid_idx] / 255.0)
            
            # Add point cloud
            scene.add_points(
                f"points/{frame_id}",
                pts_world,
                point_size=args.point_size,
                vert_color=colors,
            )
        
        print(f"  Added frame {idx + 1}/{batch['frame_num']} (original frame {frame_ids[idx]})")
    
    # Add coordinate axes
    scene.add_axes()
    
    print("\n" + "="*60)
    print("Visualization ready! Opening viewer...")
    print("="*60)
    print("Controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Use tree view on left to toggle layers")
    print("="*60)
    scene.display()

