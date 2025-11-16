# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import torch

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    Returns (0, 0) for single GPU / non-distributed training.
    """
    local_rank = os.environ.get("LOCAL_RANK", "0")
    distributed_rank = os.environ.get("RANK", "0")
    
    try:
        local_rank = int(local_rank)
        distributed_rank = int(distributed_rank)
    except (ValueError, TypeError):
        # Default to single GPU training
        local_rank = 0
        distributed_rank = 0
    
    return local_rank, distributed_rank
