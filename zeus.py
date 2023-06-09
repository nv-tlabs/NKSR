# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
This file is part of the Zeus deep learning library.
    zeus.py includes many helper functions that should be imported on top of other libraries.
"""
from typing import Optional, List

# Allow for some pre-execution hooks.
try:
    import monkey_patches
except ModuleNotFoundError:
    pass

import torch
import shutil
from pathlib import Path
from pycg import exp
from torch.nn import DataParallel
from pytorch_lightning.callbacks import Callback

# Read configuration
default_config_dir = Path(__file__).parent / "configs" / "default"
_ZEUS_OVERRIDE_CFG_PATH = Path(__file__).parent / "zeus_config.yaml"

config = exp.parse_config_yaml(default_config_dir / "zeus.yaml")
if _ZEUS_OVERRIDE_CFG_PATH.exists():
    config = exp.parse_config_yaml(_ZEUS_OVERRIDE_CFG_PATH, config)


class CopyModelFileCallback(Callback):
    """ Copy model file for the Tensorboard Logger """
    def __init__(self):
        self.source_path = None
        self.target_path = None

    def on_train_start(self, trainer, pl_module):
        if self.source_path is not None and self.target_path is not None:
            if self.target_path.parent.exists():
                shutil.move(self.source_path, self.target_path)


class CustomizedDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        inputs = self.module.module.dp_scatter(inputs, device_ids, self.dim) if inputs else []
        kwargs = self.module.module.dp_scatter(kwargs, device_ids, self.dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs


class OverfitLoggerNull:
    def __init__(self):
        self.working = False

    def log_overfit_visuals(self, *args, **kwargs):
        pass

