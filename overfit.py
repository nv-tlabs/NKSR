"""
This file is part of the Zeus deep learning library.
    test.py is mainly used to test an existing model.
"""
import zeus

import functools
import importlib
import math
import os
from pathlib import Path
from pdb import set_trace as st

import matplotlib.figure
import numpy as np
import open3d as o3d
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pycg import exp


class OverfitLogger:
    """
    Different from PL logger, this is not assigned to model.trainer.logger.
    The logging works by monkey-patch model.log_dict function to use overfit_logger instead.
    """

    def __init__(self):
        self.data = {}
        # Will include all training loss and test metrics.
        self.metrics = {}
        self.working = program_args.log_dir is not None
        self.enabled = True
        # Metrics table.
        self.metrics_df = pd.DataFrame(columns=['Step']).set_index('Step')
        self.metrics_df['Loss'] = np.nan

    def log_overfit_visuals(self, data: dict):
        if self.working and self.enabled:
            self.data.update(data)

    def log_dict(self, data: dict):
        if self.working and self.enabled:
            for k, v in data.items():
                self.metrics[k] = v

    def clear(self):
        self.data = {}
        self.metrics = {}

    def write(self, step, info_meta):
        if not self.working:
            return

        from pycg import image, render
        all_imgs = []
        for data in self.data.values():
            if isinstance(data, render.Scene):
                try:
                    all_imgs.append(data.render_opengl())
                except Exception:
                    pass
                if program_args.export:
                    for name, s_object in data.objects.items():
                        if isinstance(s_object.geom, o3d.geometry.TriangleMesh):
                            o3d.io.write_triangle_mesh(
                                str(program_args.log_dir + f"/{step:04d}.ply"), s_object.geom)
            elif isinstance(data, matplotlib.figure.Figure):
                all_imgs.append(image.from_mplot(data, close=True))

        if len(all_imgs) > 0:
            log_img = image.hlayout_images(all_imgs, background=[1.0] * 4)
            log_img = image.place_image(
                image.text(" | ".join([f"{k}: {v}" for k, v in info_meta.items()]), max_width=log_img.shape[1]),
                log_img, 0, 0)
            log_img = image.place_image(
                image.text(", ".join([f"{k}: {v:.3f}" for k, v in self.metrics.items()]), max_width=log_img.shape[1]),
                log_img, 0, 32)
            image.write(log_img, program_args.log_dir + f"/{step:04d}.png")

        # Update dataframe and export.
        # (expand new keys)
        new_keys = set(self.metrics.keys()).difference(set(self.metrics_df.columns))
        for nk in sorted(list(new_keys)):
            self.metrics_df[nk] = np.nan
        self.metrics_df.loc[step] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.metrics.items()}
        self.metrics_df['Loss'].loc[step] = info_meta['Loss']
        self.metrics_df.to_csv(program_args.log_dir + "/loss.csv")

        # matplotlib
        if program_args.log_plot:
            import matplotlib.pyplot as plt
            import warnings

            n_rows = math.ceil(len(self.metrics_df.columns) / 6)
            fig, axs = plt.subplots(figsize=(5 * 6, 5 * n_rows))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.metrics_df.plot(subplots=True, layout=(n_rows, 6), ax=axs)
            fig.savefig(program_args.log_dir + "/loss.pdf", bbox_inches='tight')
            plt.close(fig)


def to_cuda(obj):
    if isinstance(obj, tuple):
        return tuple(map(to_cuda, obj))
    if isinstance(obj, list):
        return list(map(to_cuda, obj))
    if isinstance(obj, dict):
        return dict(map(to_cuda, obj.items()))
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    return obj


def overfit_log_dict(dictionary, self, *args, **kwargs):
    self.overfit_logger.log_dict(dictionary)


@exp.mem_profile
def run_overfit():
    # reporter = MemReporter(net_model)

    # Peek data before even initialize parameters.
    net_model.on_before_overfit(ofit_data)
    net_model.cuda()

    optimizers, schedulers = net_model.configure_optimizers()
    assert len(optimizers) == 1 and len(schedulers) == 1

    optimizer, scheduler = optimizers[0], schedulers[0]
    assert scheduler['interval'] == 'step'
    scheduler = scheduler['scheduler']

    for step in range(program_args.steps):
        if step % program_args.interval == 0:
            print("--- Testing...")
            net_model.eval()
            net_model.overfit_logger.clear()
            with torch.no_grad():
                net_model.test_step(ofit_data, 0)
            # Pause recording to avoid training logs to come in...
            #   We can write here, but the loss will be missing...
            net_model.overfit_logger.enabled = False

        # reporter.report()

        optimizer.zero_grad()
        net_model.train()
        loss = net_model.training_step(ofit_data, 0)
        loss.backward()
        net_model.on_after_backward()
        optimizer.step()
        scheduler.step()

        if 'MEM_PROFILE' in os.environ.keys():
            st()

        if step % program_args.interval == 0:
            net_model.overfit_logger.enabled = True
            net_model.overfit_logger.write(step, {'Step': step, 'Loss': loss.item()})
            net_model.overfit_logger.clear()

        print(f"--- Step {step}. Loss = {loss.item():.2f}, LR = {scheduler.get_last_lr()}.")


if __name__ == '__main__':
    pl.seed_everything(0)

    # Get model arguments.
    program_parser = exp.argparse.ArgumentParser()
    program_parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    program_parser.add_argument('--steps', type=int, default=20000, help='Number of overfitting steps.')
    program_parser.add_argument('--interval', type=int, default=1, help='Perform test step every interval steps.')
    program_parser.add_argument('--data', type=str, default='test-0',
                                help='<train/test/val>-<id> of the data you want to overfit.')
    program_parser.add_argument('--log_dir', type=str, default=None, help='Path to log')
    program_parser.add_argument('--log_plot', action='store_true', help='Export loss curves.')
    program_parser.add_argument('--export', action='store_true', help='Export meshes if exists.')

    program_args, other_args = program_parser.parse_known_args()
    model_parser = exp.ArgumentParserX(base_config_path=zeus.default_config_dir / "train.yaml")
    model_args = model_parser.parse_args(other_args)
    hyper_path = model_args.hyper
    del model_args["hyper"]

    # Force visualization if we want to dump logging
    if program_args.log_dir is not None:
        model_args.visualize = True
        Path(program_args.log_dir).mkdir(parents=True, exist_ok=True)

    # Get model prepared
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)
    net_model = net_model.cuda()
    print(OmegaConf.to_yaml(net_model.hparams, resolve=True))

    # Mock a trainer.
    ofit_trainer = OmegaConf.create()
    ofit_trainer.world_size = 1
    ofit_trainer.logger = None
    ofit_trainer.global_step = 0
    ofit_trainer.training = True
    net_model.trainer = ofit_trainer
    net_model.overfit_logger = OverfitLogger()
    net_model.log_dict = functools.partial(overfit_log_dict, self=net_model)

    # Prepare data
    od_split, od_idx = program_args.data.split('-')
    dataloader = getattr(net_model, od_split + "_dataloader")()
    ofit_trainer.dataset_short_name = dataloader.dataset.get_short_name()
    ofit_data = None
    for od_cur, ofit_data in enumerate(dataloader):
        if od_cur == int(od_idx):
            break
    ofit_data = to_cuda(ofit_data)

    # Start overfitting
    with exp.AutoPdb():
        run_overfit()
