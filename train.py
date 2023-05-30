# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
This file is part of the Zeus deep learning library.
    train.py is mainly used to start a long training process (probably on server).
"""
import zeus
import bdb
import importlib
import randomname
import os
import pdb
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

import pytorch_lightning as pl
from datetime import datetime, timedelta
import torch
import uuid
import yaml
from omegaconf import OmegaConf
from pycg import exp, wdb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def determine_usable_gpus():
    if program_args.gpus is None:
        program_args.gpus = 1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        original_cvd = [int(t) for t in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        original_cvd = []

    if len(original_cvd) == program_args.gpus:
        # Everything is fine.
        return

    # Mismatched/missing CVD setting & #gpus, reset.
    gpu_states = exp.get_gpu_status("localhost")
    available_gpus = [t for t in gpu_states if t.gpu_mem_usage < 0.2 and t.gpu_compute_usage < 0.2]

    if len(available_gpus) == 0:
        exp.logger.fatal("You cannot use GPU. Everything is full.")
        sys.exit(0)

    if len(available_gpus) < program_args.gpus:
        print(f"Warning: Available GPUs are {[t.gpu_id for t in available_gpus]}, "
              f"but you want to use {program_args.gpus} GPUs.")
        program_args.gpus = len(available_gpus)

    available_gpus = available_gpus[:program_args.gpus]
    selection_str = ','.join([str(t.gpu_id) for t in available_gpus])
    exp.logger.info(f"Intelligent GPU selection: {selection_str}")
    os.environ['CUDA_VISIBLE_DEVICES'] = selection_str

    if program_args.gpus > 1 and program_args.accelerator is None:
        program_args.accelerator = 'ddp'


def is_rank_zero():
    # It will also set LOCAL_RANK env variable, so using that will be more consistent.
    return os.environ.get('MASTER_PORT', None) is None


def remove_option(parser, option):
    for action in parser._actions:
        if vars(action)['option_strings'][0] == option:
            parser._handle_conflict_resolve(None, [(option, action)])
            break


def readable_name_from_exec(exec_list: List[str]):
    """ Convert --exec configs into readable names (mainly used in wandb sweeps) """
    keys = {}
    for exec_str in exec_list:
        kvs = exec_str.split("=")
        k_name = kvs[0]
        k_name_arr = ["".join([us[0] for us in t.split("_") if len(us) > 0]) for t in k_name.split(".")]
        # Collapse leading dots except for the last one.
        k_name = ''.join(k_name_arr[:-2]) + '.'.join(k_name_arr[-2:])
        k_value = kvs[1]
        if k_value.lower() in ["true", "false"]:
            k_value = str(int(k_value.lower() == "true"))
        keys[k_name] = k_value
    return '-'.join([k + keys[k] for k in sorted(list(keys.keys()))])


if __name__ == '__main__':
    """""""""""""""""""""""""""""""""""""""""""""""
    [1] Parse and initialize program arguments
        these include: --debug, --profile, --gpus, --num_nodes, --resume, ...
        they will NOT be saved for a checkpoints.
    """""""""""""""""""""""""""""""""""""""""""""""
    program_parser = exp.argparse.ArgumentParser()
    program_parser.add_argument('--debug', action='store_true', help='Use debug mode of pytorch')
    program_parser.add_argument('--resume', action='store_true', help='Continue training. Use hparams.yaml file.')
    program_parser.add_argument('--nolog', action='store_true', help='Do not create any logs.')
    program_parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    program_parser.add_argument('--save_topk', default=1, type=int,
                                help='How many top models to save. -1 to save all models.')
    program_parser.add_argument('--validate_first', action='store_true',
                                help='Do a full validation with logging before training starts.')
    program_parser.add_argument('--logger_type', choices=['tb', 'wandb', 'none'], default='wandb')
    program_parser.add_argument('--wname', default=None, type=str, help='Run name to be appended to wandb')

    program_parser = pl.Trainer.add_argparse_args(program_parser)
    # Remove some args, which we think should be model-based.
    remove_option(program_parser, '--accumulate_grad_batches')
    program_args, other_args = program_parser.parse_known_args()

    # Force not to sync to shorten bootstrap time.
    if program_args.nosync:
        os.environ['NO_SYNC'] = '1'

    # Train forever
    if program_args.max_epochs is None:
        program_args.max_epochs = -1

    # Default logger type
    if program_args.nolog:
        program_args.logger_type = 'none'

    if is_rank_zero():
        # Detect usable GPUs.
        determine_usable_gpus()
    else:
        # Align parameters.
        program_args.accelerator = 'ddp'

    # Profiling and debugging options
    torch.autograd.set_detect_anomaly(program_args.debug)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        save_top_k=program_args.save_topk,
        every_n_epochs=1,
        mode='min',
    )
    lr_record_callback = LearningRateMonitor(logging_interval='step')
    copy_model_file_callback = zeus.CopyModelFileCallback()

    """""""""""""""""""""""""""""""""""""""""""""""
    [2] Determine model arguments
        MODEL args include: --lr, --num_layers, etc. (everything defined in YAML)
        These use AP-X module, which accepts CLI and YAML inputs.
        These args will be saved as hyper-params.
    """""""""""""""""""""""""""""""""""""""""""""""
    if program_args.resume:
        raw_hyper = other_args[0]
        if raw_hyper.startswith("wdb:"):
            # Load config and replace
            wdb_run, wdb_ckpt = wdb.get_wandb_run(raw_hyper, zeus.config.wandb.base, default_ckpt="last")
            tmp_yaml_name = '/tmp/' + str(uuid.uuid4()) + '.yaml'
            with open(tmp_yaml_name, 'w') as outfile:
                yaml.dump(wdb.recover_from_wandb_config(wdb_run.config), outfile)
            other_args[0] = tmp_yaml_name

    model_parser = exp.ArgumentParserX(base_config_path=zeus.default_config_dir / "train.yaml")
    model_args = model_parser.parse_args(other_args)
    hyper_path = model_args.hyper
    del model_args["hyper"]

    """""""""""""""""""""""""""""""""""""""""""""""
    [3] Build / restore logger and checkpoints.
    """""""""""""""""""""""""""""""""""""""""""""""
    # Set checkpoint auto-save options.
    last_ckpt_path = None
    if program_args.logger_type == 'tb':
        if not program_args.resume:
            logger_version_num = None
            last_ckpt_path = None
        else:
            # Resuming stuff.
            last_ckpt_path = Path(hyper_path).parent / "checkpoints" / "last.ckpt"
            logger_version_num = Path(hyper_path).parent.name if program_args.resume else None
        logger = TensorBoardLogger(zeus.config.tb.base, name=model_args.name,
                                   version=logger_version_num, default_hp_metric=False)
        # Call this property to assign the version early, so we don't have to wait for the model to be loaded
        print(f"Tensorboard logger, version number =", logger.version)
    elif program_args.logger_type == 'wandb':
        assert len(zeus.config.wandb.user) > 0, "Please setup wandb user!"
        # Will create wandb folder automatically
        if not program_args.resume:
            wname = program_args.wname
            if 'WANDB_SWEEP_ID' in os.environ.keys():
                # (Use exec to determine good names)
                wname = os.environ['WANDB_SWEEP_ID'] + "-" + readable_name_from_exec(model_args.exec)
            if wname is None:
                # Example: 0105-clever-monkey
                wname = (datetime.utcnow() + timedelta(hours=8)).strftime('%m%d') + "-" + randomname.get_name()
            sep_pos = str(model_args.name).find('/')
            if sep_pos == -1:
                project_name = model_args.name
                run_name = "root/" + wname
            else:
                project_name = model_args.name[:sep_pos]
                run_name = model_args.name[sep_pos + 1:] + "/" + wname
            logger = WandbLogger(name=run_name, save_dir=zeus.config.wandb.base, project='nkfw-' + project_name)
        else:
            logger = WandbLogger(name=wdb_run.name, save_dir=zeus.config.wandb.base, project=wdb_run.project, id=wdb_run.id)
            last_ckpt_path = wdb_ckpt
            os.unlink(tmp_yaml_name)

    else:
        logger = None

    # Copy model file to a temporary location.
    if is_rank_zero() and not program_args.resume:
        if program_args.logger_type == 'tb':
            trainer_log_dir = logger.log_dir
            temp_py_path = Path(tempfile._get_default_tempdir()) / next(tempfile._get_candidate_names())
            shutil.copy(f"models/{model_args.model.replace('.', '/')}.py", temp_py_path)
            copy_model_file_callback.source_path = temp_py_path
            copy_model_file_callback.target_path = Path(trainer_log_dir) / "model.py"
        elif program_args.logger_type == 'wandb':
            import wandb

            def should_save(path):
                relpath = os.path.relpath(path, os.getcwd())
                return (relpath.startswith("models/") or relpath.startswith("dataset/")) and relpath.endswith(".py")

            wandb.run.log_code(include_fn=should_save)

    """""""""""""""""""""""""""""""""""""""""""""""
    [4] Build trainer and determine final hparams
    """""""""""""""""""""""""""""""""""""""""""""""
    # Do it here because wandb name need some randomness.
    pl.seed_everything(0)

    # Build trainer
    trainer = pl.Trainer.from_argparse_args(
        program_args,
        callbacks=[checkpoint_callback, lr_record_callback, copy_model_file_callback]
        if logger is not None else [],
        logger=logger,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        auto_select_gpus=True,
        accumulate_grad_batches=model_args.accumulate_grad_batches)
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)

    if is_rank_zero():
        print(" >>>> ======= MODEL HYPER-PARAMETERS ======= <<<< ")
        print(OmegaConf.to_yaml(net_model.hparams, resolve=True))
        print(" >>>> ====================================== <<<< ")

    # No longer use this..
    del is_rank_zero

    """""""""""""""""""""""""""""""""""""""""""""""
    [5] Main training iteration.
    """""""""""""""""""""""""""""""""""""""""""""""
    # Note: In debug mode, trainer.fit will automatically end if NaN occurs in backward.
    e = None
    try:
        net_model.overfit_logger = zeus.OverfitLoggerNull()
        if program_args.validate_first:
            trainer.validate(net_model, ckpt_path=last_ckpt_path)
        with exp.pt_profile_named("training", "train.json"):
            trainer.fit(net_model, ckpt_path=last_ckpt_path)
    except Exception as ex:
        e = ex
        # https://stackoverflow.com/questions/52081929/pdb-go-to-a-frame-in-exception-within-exception
        if isinstance(e, MisconfigurationException):
            if e.__context__ is not None:
                traceback.print_exc()
                if program_args.accelerator is None:
                    pdb.post_mortem(e.__context__.__traceback__)
        elif isinstance(e, bdb.BdbQuit):
            exp.logger.info("Post mortem is skipped because the exception is from Pdb.")
        else:
            traceback.print_exc()
            if program_args.accelerator is None:
                pdb.post_mortem(e.__traceback__)

    """""""""""""""""""""""""""""""""""""""""""""""
    [6] If ended premature, add to delete list.
    """""""""""""""""""""""""""""""""""""""""""""""
    if trainer.current_epoch < 1 and last_ckpt_path is None and trainer.local_rank == 0:
        if program_args.logger_type == 'tb':
            if Path(trainer.log_dir).exists():
                with open(".premature_checkpoints", "a") as f:
                    f.write(f"{trainer.log_dir}\n")
                exp.logger.info(f"\n\nTB Checkpoint at {trainer.log_dir} marked to be cleared.\n\n")
            sys.exit(-1)
        elif program_args.logger_type == 'wandb':
            with open(".premature_checkpoints", "a") as f:
                f.write(f"wdb:{trainer.logger.experiment.path}:{trainer.logger.experiment.name}\n")
            exp.logger.info(f"\n\nWandb Run of {trainer.logger.experiment.path} "
                  f"(with name {trainer.logger.experiment.name}) marked to be cleared.\n\n")
            sys.exit(-1)

    if trainer.local_rank == 0:
        exp.logger.info(f"Training Finished. Best path = {checkpoint_callback.best_model_path}")
