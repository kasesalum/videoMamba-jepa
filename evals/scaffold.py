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


def _eval_module(eval_name, args_eval):
    model_name = args_eval.get('pretrain', {}).get('model_name', '')
    if model_name.startswith('videomamba'):
        return f'evals.{eval_name}.eval_videomamba'
    return f'evals.{eval_name}.eval'


def main(
    eval_name,
    args_eval,
    resume_preempt=False
):
    logger.info(f'Running evaluation: {eval_name}')
    model_name = (
        args_eval.get('pretrain', {}).get('model_name', '')
        if isinstance(args_eval, dict) else ''
    )
    eval_module = 'eval_videomamba' if model_name.startswith('videomamba') else 'eval'
    return importlib.import_module(f'evals.{eval_name}.{eval_module}').main(
        args_eval=args_eval,
        resume_preempt=resume_preempt)
