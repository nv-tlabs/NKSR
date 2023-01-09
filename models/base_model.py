"""
This file is part of the Zeus deep learning library.
    base_model.py provides a BaseModel to be inherited for implementing different models.
"""
import zeus

import functools
import gc
import importlib
import inspect
import pickle
import shutil
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Any, Optional
from datetime import datetime

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import omegaconf.errors
import open3d as o3d
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pycg import exp, image
from pycg.exp import AverageMeter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.summary import hparams


exp.global_var_manager.register_variable('skip_backward', False)


def lambda_lr_wrapper(it, lr_config, batch_size):
    """ Outside to make DDP happy """
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.best_metrics = AverageMeter()

        # For recording test information
        # step -> log_name -> log_value (list of ordered-dict)
        self.test_logged_values = []
        self.record_folder = None
        self.record_headers = []
        self.record_data_cache = {}
        self.last_test_valid = False

        # Track the number of OOMs.
        self.num_oom = 0

    @staticmethod
    def load_module(spec_path, weight_path=None, overwrite_config=None):
        """
        Load a module given spec_path
        :param spec_path: Path to a model yaml file or ckpt. If is a ckpt file, then weight will also be loaded.
        :param weight_path: Path to the model weight. If explicitly set to 'NO_LOAD', then even if ckpt is provided to
            spec_path, no weights will be loaded into the model.
        :param overwrite_config: argparse.Namespace object, if you want to overwrite the original config.
        :return: the module class, possibly with weight loaded.
        """
        if spec_path is not None:
            spec_path = Path(spec_path)
            if spec_path.suffix == ".ckpt":
                # Boil down spec path using glob.
                import glob2
                possible_paths = glob2.glob(str(spec_path))
                if len(possible_paths) == 1:
                    spec_path = Path(possible_paths[0])
                else:
                    raise AssertionError
                config_yaml_path = spec_path.parent.parent / "hparams.yaml"
                if weight_path == "NO_LOAD":
                    weight_path = None
                elif weight_path is None:
                    weight_path = spec_path
            elif spec_path.suffix == ".yaml":
                config_yaml_path = spec_path
            else:
                raise NotImplementedError

            config_args = exp.parse_config_yaml(config_yaml_path, overwrite_config, override=False)
        else:
            config_args = overwrite_config

        if "model" not in config_args.keys():
            print("No model found.")
            return None

        basis_net_module = importlib.import_module("models." + config_args.model).Model

        if weight_path is not None:
            net_module = basis_net_module.load_from_checkpoint(weight_path, hparams=config_args)
        else:
            net_module = basis_net_module(config_args)

        return net_module

    @property
    def logger_type(self):
        logger = self.trainer.logger
        if logger is None:
            return 'none'
        elif 'Wandb' in type(logger).__name__:
            return 'wandb'
        else:
            return 'tb'

    def get_dataset_spec(self):
        raise NotImplementedError

    def get_collate_fn(self):
        raise NotImplementedError

    def get_hparams_metrics(self):
        raise NotImplementedError

    def train_val_step(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        try:
            return self.train_val_step(is_val=False, *args, **kwargs)
        except RuntimeError:
            # Compare to post-mortem, this would allow training to continue...
            exp.logger.warning(f"Training-step OOM. Skipping.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            self.num_oom += 1.0
            self.log("num_oom", self.num_oom)
            return None

    def validation_step(self, *args, **kwargs):
        try:
            return self.train_val_step(is_val=True, *args, **kwargs)
        except RuntimeError:
            # Compare to post-mortem, this would allow training to continue...
            exp.logger.warning(f"Validation-step OOM. Skipping.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            self.num_oom += 1.0
            self.log("num_oom", self.num_oom)
            return None

    def on_before_overfit(self, batch):
        pass

    def tprint(self, *args):
        torch.set_printoptions(sci_mode=False, linewidth=200)
        if self.trainer is None or self.trainer.testing:
            print(*args)
        torch.set_printoptions(profile="default")

    def configure_optimizers(self):
        """
        Some comments the optimizers and schedulers.
            Usually just try SGD and Adam(W) will be fine, the latter of which does not depend too much on the
        learning rate. Other (RMSProp, AdaGrad, AdaDelta are bad).
            For schedulers, we use an approximation of exponential decay. Other strategies like polynomial or inverse
        square root are just alike, so we don't need to try them.
        """
        lr_config = self.hparams.learning_rate
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        exp.global_var_manager.set('skip_backward', False)

    def on_after_backward(self):
        # You can do gradient checking and clipping here. Directly on the models.
        # ... Haven't check what will happen for DDP though...
        if exp.global_var_manager.get('skip_backward'):
            exp.logger.info("Skip backward is enabled. This step won't affect the model.")
            for p in filter(lambda p: p.grad is not None, self.parameters()):
                p.grad.data.zero_()
            exp.global_var_manager.set('skip_backward', False)
            return

        # The following inspections can be added to logger by using --track_grad_norm 2.0
        grad_clip_val = self.hparams.get('grad_clip', 1000.)

        if grad_clip_val == "inspect":
            from pytorch_lightning.utilities.grads import grad_norm
            grad_dict = grad_norm(self, 'inf')      # Get the maximum absolute value.
            print(grad_dict)
            grad_clip_val = 1000.

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_val)
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=grad_clip_val)

        # If detect nan values, then this step is skipped
        has_nan_value_cnt = 0
        for p in filter(lambda p: p.grad is not None, self.parameters()):
            if torch.any(p.grad.data != p.grad.data):
                has_nan_value_cnt += 1
        if has_nan_value_cnt > 0:
            exp.logger.warning(f"{has_nan_value_cnt} parameters get nan-gradient -- this step will be skipped.")
            for p in filter(lambda p: p.grad is not None, self.parameters()):
                p.grad.data.zero_()

    def on_fit_start(self):
        if self.trainer.logger is None:
            return
        if self.logger_type == 'tb':
            writer = self.trainer.logger.experiment._get_file_writer()
            if writer is not None:
                hparams_metrics = {}
                for metric_name, should_add_best in self.get_hparams_metrics():
                    hparams_metrics[metric_name] = 0.0
                    hparams_metrics[f'best/{metric_name}'] = 0.0
                from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict
                hparams_dict = _convert_params(self.hparams)
                hparams_dict = _flatten_dict(hparams_dict)
                hparams_dict = self.trainer.logger._sanitize_params(hparams_dict)
                exp, ssi, sei = hparams(hparams_dict, hparams_metrics)
                writer.add_summary(exp)
                writer.add_summary(ssi)
                writer.add_summary(sei)
                # Note: only the name of the metrics, instead of their values (0.0) will be written.

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            logged_metrics = self.trainer.callback_metrics
            for metric_name, should_add_best in self.get_hparams_metrics():
                if should_add_best and metric_name in logged_metrics.keys():
                    self.best_metrics.append_loss({metric_name: logged_metrics[metric_name].item()})
                    # metric in hparams in tensorboard will automatically update according to added scalars.
                    #   so we do not need to manually update hyperparams.
                    self.log(f'best/{metric_name}', np.min(self.best_metrics.loss_dict[metric_name]))

    def log(self, name: str, value: Any, *args, **kwargs):
        """
            Override PL's default log function, to better support test-time or overfit-time no-logger logging
        (using various types including string).
        """
        # For overfitting, we leave the log to the log_dict monkey_patch.
        if isinstance(self.trainer, omegaconf.dictconfig.DictConfig):
            return
        if self.trainer.testing:
            if name == 'batch-idx':
                self.test_logged_values.append(OrderedDict([(name, value)]))
            else:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.test_logged_values[-1][name] = value
        else:
            super().log(name=name, value=value, *args, **kwargs)

    def log_dict_prefix(
        self,
        prefix: str,
        dictionary: Mapping[str, Any],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None
    ):
        """
        This overrides fixes if dict key is not a string...
        """
        dictionary = {
            prefix + "/" + str(k): v for k, v in dictionary.items()
        }
        self.log_dict(dictionary=dictionary,
                      prog_bar=prog_bar,
                      logger=logger, on_step=on_step, on_epoch=on_epoch)

    def log_image(self, name: str, img: np.ndarray):
        if self.trainer.logger is not None:
            if self.logger_type == 'tb':
                if img.shape[2] <= 4:
                    # WHC -> CWH
                    img = np.transpose(img, (2, 0, 1))
                self.trainer.logger.experiment.add_image(name, img, self.trainer.global_step)
            elif self.logger_type == 'wandb':
                self.trainer.logger.log_image(key=name, images=[img])

    def log_plot(self, name: str, fig: matplotlib.figure.Figure, close: bool = True):
        img = image.from_mplot(fig, close)
        self.log_image(name, img)
        if close:
            plt.close(fig)

    def log_geometry(self, name: str, geom, draw_color: bool = False):
        if self.trainer.logger is None:
            return
        if isinstance(geom, o3d.geometry.TriangleMesh):
            from pycg import render
            mv_img = render.multiview_image(
                [geom], viewport_shading='LIT' if draw_color else 'NORMAL', backend='filament')
            self.log_image("mesh" + name, mv_img)
        else:
            raise NotImplementedError

    def test_log_data(self, data_dict: dict):
        # Output artifact data only when there is no focus.
        if self.record_folder is None or self.hparams.focus == "none":
            return
        self.record_data_cache.update(data_dict)

    def get_dataset_short_name(self):
        # Used for identify, e.g., camera paths.
        try:
            return self.trainer.test_dataloaders[0].dataset.get_short_name()
        except omegaconf.errors.ConfigAttributeError:
            return self.trainer.dataset_short_name

    def on_test_start(self):
        if self.hparams.get('record', None) is not None:
            test_set_name = self.trainer.test_dataloaders[0].dataset.get_name()
            if len(self.hparams.record) == 0:
                model_name = inspect.getfile(self.__class__).split('/')[-1].split('.py')[0]
                cur_time = datetime.now().strftime("%b%d-%X")
                save_folder_name = f"{cur_time}-{model_name}"
            else:
                save_folder_name = self.hparams.record[0]
            self.record_folder = Path(zeus.config.test_path) / test_set_name / save_folder_name
            exp.mkdir_confirm(self.record_folder)
            exp.logger.info(f"Records will be saved at {self.record_folder}")
            with (self.record_folder / "hparams.yaml").open('w') as f:
                # yaml.dump(dict(self.hparams), f)
                OmegaConf.save(self.hparams, f)
            shutil.copy(f"models/{self.hparams.model.replace('.', '/')}.py", self.record_folder / "model.py")

        # Monkey-patch s.t. we can specify one batch to run on.
        if self.hparams.focus != "none":
            old_test_step = self.test_step
            def focus_test_step(batch, batch_idx):
                self.last_test_valid = False
                if self.hparams.focus != "all":
                    if self.hparams.focus[0] == "g":    # greater than
                        if batch_idx <= int(self.hparams.focus[1:]):
                            return
                    elif self.hparams.focus[0] == "l":  # less than
                        if batch_idx >= int(self.hparams.focus[1:]):
                            return
                    elif ',' in self.hparams.focus:
                        focus_ids = [int(t) for t in self.hparams.focus.split(',')]
                        if batch_idx not in focus_ids:
                            return
                    else:
                        if batch_idx != int(self.hparams.focus):
                            return
                old_test_step(batch, batch_idx)
                self.last_test_valid = True

            self.test_step = focus_test_step
        else:
            self.last_test_valid = True

    def print_test_logs(self):
        sample_log = self.test_logged_values[0]
        headers = list(sample_log.keys())
        headers.remove('batch-idx')
        print("Test logs:")
        for h in headers:
            header_values = [t[h] for t in self.test_logged_values if h in t]
            print(h, f"({len(header_values)})",
                  np.mean(header_values) if isinstance(header_values[0], float) else "NOT-NUMERIC")

    # Defines special data that requires separate folder to save
    TEST_SPECIAL_META = {
        o3d.geometry.TriangleMesh: ["ply", o3d.io.write_triangle_mesh],
        o3d.geometry.PointCloud: ["ply", o3d.io.write_point_cloud],
        dict: ["pth", lambda f, data: torch.save(data, f)],
        np.ndarray: ["npy", np.save]
    }

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        self.log('batch-idx', batch_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if not self.last_test_valid:
            return

        if self.record_folder is not None:
            # If has valid log
            if len(self.test_logged_values) > 0 and self.test_logged_values[-1]['batch-idx'] == batch_idx:
                cur_log = self.test_logged_values[-1]
                csv_path = self.record_folder / "result.csv"
                if len(self.record_headers) == 0:
                    self.record_headers = list(cur_log.keys())
                    self.record_headers.remove('batch-idx')
                    self.record_headers.insert(0, 'batch-idx')
                    with csv_path.open('w') as f:
                        f.write(",".join(self.record_headers) + "\n")
                with csv_path.open('a') as f:
                    # Cur log can be of other type:
                    value_list = []
                    for t in self.record_headers:
                        if t in cur_log.keys():
                            cur_log_t = cur_log[t]
                            value_list.append(str(cur_log_t))
                        else:
                            value_list.append("-")
                    f.write(",".join(value_list) + "\n")

            # Write output data if any.
            if len(self.record_data_cache) > 0:

                pkl_data = {k: v for k, v in self.record_data_cache.items()
                            if type(v) not in self.TEST_SPECIAL_META.keys()}
                special_data = {k: v for k, v in self.record_data_cache.items() if k not in pkl_data.keys()}

                for k, v in special_data.items():
                    (self.record_folder / k).mkdir(exist_ok=True, parents=True)
                    suffix, write_f = self.TEST_SPECIAL_META[type(v)]
                    write_f(str(self.record_folder / k / f"{batch_idx:06d}.{suffix}"), v)

                if len(pkl_data) > 0:
                    (self.record_folder / "test_log_data").mkdir(exist_ok=True, parents=True)

                    def convert_tensor(obj):
                        if isinstance(obj, tuple):
                            return tuple(map(convert_tensor, obj))
                        if isinstance(obj, list):
                            return list(map(convert_tensor, obj))
                        if isinstance(obj, dict):
                            return dict(map(convert_tensor, obj.items()))
                        if isinstance(obj, torch.Tensor):
                            return obj.detach().cpu().numpy()
                        return obj

                    try:
                        res = convert_tensor(pkl_data)
                    finally:
                        convert_tensor = None

                    with (self.record_folder / "test_log_data" / f"{batch_idx:06d}.pkl").open("wb") as f:
                        pickle.dump(res, f)

        self.record_data_cache = {}

    def dp_scatter(self, inputs, target_gpus, dim=0):
        """
        Define how dp scatter a batch of data. This only takes effect when using multiple GPUs.
        """
        from torch.nn.parallel.scatter_gather import scatter
        return scatter(inputs, target_gpus, dim)

    def _determine_batch_size(self):
        raise NotImplementedError

    def train_dataloader(self):
        # Note:
        import dataset
        train_set = dataset.build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs)
        torch.manual_seed(0)
        return DataLoader(train_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=True,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        import dataset
        val_set = dataset.build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)
        return DataLoader(val_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=False,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        import dataset
        test_set = dataset.build_dataset(
            self.hparams.test_dataset, self.get_dataset_spec(), self.hparams, self.hparams.test_kwargs)
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        return DataLoader(test_set, batch_size=self.hparams.batch_size // self.trainer.world_size,
                          shuffle=self.hparams.test_set_shuffle,
                          num_workers=self.hparams.test_num_workers, collate_fn=self.get_collate_fn())
