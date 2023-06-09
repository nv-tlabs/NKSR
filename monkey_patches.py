# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import traceback
import warnings
import sys
import os

"""
Get rid of tensorboard warnings.
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
Warning:
    Inspect where warning happens.
"""
if False:
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
        import pdb; pdb.set_trace()
    warnings.showwarning = warn_with_traceback


"""
PL's batch size cannot be correctly computed in NKSR, fix it.
"""
if True:
    import pytorch_lightning as pl
    # Monkey-patch `extract_batch_size` to not raise warning from weird tensor sizes

    def extract_bs(self, *args, **kwargs):
        batch_size = 1
        self.batch_size = batch_size
        return batch_size

    pl.trainer.connectors.logger_connector.result._ResultCollection._extract_batch_size = extract_bs
