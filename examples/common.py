# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import requests
from pathlib import Path
from pycg import vis, exp
from pyntcloud import PyntCloud


DOWNLOAD_URL = "https://nksr.huangjh.tech"


def load_bunny_example():
    bunny_path = Path(__file__).parent.parent / "assets" / "bunny.ply"
    bunny_geom = vis.from_file(bunny_path)
    return bunny_geom


def load_spot_example():
    spot_path = Path(__file__).parent.parent / "assets" / "spot.ply"
    spot_geom = vis.from_file(spot_path)
    return spot_geom


def load_buda_example():
    buda_path = Path(__file__).parent.parent / "assets" / "buda.ply"

    if not buda_path.exists():
        exp.logger.info("Downloading assets...")
        res = requests.get(f"{DOWNLOAD_URL}/buda.ply")
        with open(buda_path, "wb") as f:
            f.write(res.content)
        exp.logger.info("Download finished!")

    buda_geom = vis.from_file(buda_path)
    buda_geom.scale(50.0, center=np.zeros(3))
    return buda_geom


def load_scannet_example():
    scannet_path = Path(__file__).parent.parent / "assets" / "scannet.ply"

    if not scannet_path.exists():
        exp.logger.info("Downloading assets...")
        res = requests.get("{DOWNLOAD_URL}/scannet-rgbd.ply")
        with open(scannet_path, "wb") as f:
            f.write(res.content)
        exp.logger.info("Download finished!")

    scannet_geom = vis.from_file(scannet_path)
    return scannet_geom


def load_waymo_example():
    waymo_path = Path(__file__).parent.parent / "assets" / "waymo-pcd.ply"

    if not waymo_path.exists():
        exp.logger.info("Downloading assets...")
        res = requests.get("{DOWNLOAD_URL}/waymo-pcd.ply")
        with open(waymo_path, "wb") as f:
            f.write(res.content)
        exp.logger.info("Download finished!")

    pcloud = PyntCloud.from_file(str(waymo_path))
    pdata = pcloud.points
    xyz = np.stack([pdata['x'], pdata['y'], pdata['z']], axis=1)
    sensor = np.stack([pdata['sensor_x'], pdata['sensor_y'], pdata['sensor_z']], axis=1)
    return xyz, sensor


def warning_on_low_memory(threshold_mb: float):
    gpu_status = exp.get_gpu_status('localhost')
    if len(gpu_status) == 0:
        exp.logger.fatal("No GPU found!")
        return

    gpu_status = gpu_status[0]
    available_mb = (gpu_status.gpu_mem_total - gpu_status.gpu_mem_byte) / 1024. / 1024.

    if available_mb < threshold_mb:
        exp.logger.warning("Available GPU memory is {:.2f} MB, "
                           "we recommend you to have more than {:.2f} MB available.".format(available_mb, threshold_mb))
