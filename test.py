"""
This file is part of the Zeus deep learning library.
    test.py is mainly used to test an existing model.
"""

import zeus
import bdb
import os

import omegaconf

import importlib
import argparse
from pycg import exp, wdb
import pytorch_lightning as pl
from pathlib import Path


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser


if __name__ == '__main__':
    pl.seed_everything(0)

    parser = exp.ArgumentParserX(base_config_path=zeus.default_config_dir / 'test.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt', type=str, required=False, help='Path to ckpt file.')
    parser.add_argument('--weight', type=str, required=False, default='default',
                        help="Overwrite the weight defined by --ckpt. "
                             "Explicitly set to 'none' so that no weight will be loaded.")
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--record', nargs='*',
                        help='Whether or not to store evaluation data. add name to specify save path.')
    parser.add_argument('--focus', type=str, default="none", help='Sample to focus')

    known_args = parser.parse_known_args()[0]
    args_ckpt = known_args.ckpt

    if args_ckpt is not None:
        if args_ckpt.startswith("wdb:"):
            wdb_run, args_ckpt = wdb.get_wandb_run(args_ckpt, wdb_base=zeus.config.wandb.base, default_ckpt="test_auto")
            assert args_ckpt is not None, "Please specify checkpoint version!"
            assert args_ckpt.exists(), "Selected checkpoint does not exist!"
            model_args = omegaconf.OmegaConf.create(wdb.recover_from_wandb_config(wdb_run.config))
        else:
            model_yaml_path = Path(known_args.ckpt).parent.parent / "hparams.yaml"
            model_args = exp.parse_config_yaml(model_yaml_path)
    else:
        model_args = None
    args = parser.parse_args(additional_args=model_args)

    if args.nosync:
        # Force not to sync to shorten bootstrap time.
        os.environ['NO_SYNC'] = '1'

    if args.gpus is None:
        args.gpus = 1

    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**args), logger=None, max_epochs=1)
    net_module = importlib.import_module("models." + args.model).Model

    # --ckpt & --weight logic:
    if args.weight == 'default':
        ckpt_path = args_ckpt
    elif args.weight == 'none':
        ckpt_path = None
    else:
        ckpt_path = args.weight

    try:
        if ckpt_path is not None:
            net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args)
        else:
            net_model = net_module(args)
        net_model.overfit_logger = zeus.OverfitLoggerNull()

        with exp.pt_profile_named("trainer.test", "test.json"):
            test_result = trainer.test(net_model)

        # Usually, PL will output aggregated test metric from LoggerConnector (obtained from trainer.results)
        #   However, as we patch self.log for test. We would print that ourselves.
        net_model.print_test_logs()

    except Exception as ex:
        if isinstance(ex, bdb.BdbQuit):
            exp.logger.info("Post mortem is skipped because the exception is from Pdb. Bye!")
        elif isinstance(ex, KeyboardInterrupt):
            exp.logger.info("Keyboard Interruption. Program end normally.")
        else:
            import sys, pdb, traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
            sys.exit(-1)
