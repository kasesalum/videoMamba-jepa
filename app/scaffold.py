# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _pretrain_module(app, args):
    model_name = args.get('model', {}).get('model_name', '')
    if model_name.startswith('videomamba'):
        return f'app.{app}.train_videomamba'
    return f'app.{app}.train'


def main(app, args, resume_preempt=False):

    logger.info(f'Running pre-training of app: {app}')
    trainer = args.get('trainer')
    if trainer is None:
        model_name = args.get('model', {}).get('model_name', '')
        trainer = 'train_videomamba' if model_name.startswith('videomamba') else 'train'
    logger.info(f'Using trainer module: app.{app}.{trainer}')
    return importlib.import_module(f'app.{app}.{trainer}').main(
        args=args,
        resume_preempt=resume_preempt)
