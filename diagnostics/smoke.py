import argparse
import copy
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa.diagnostic_utils import (
    apply_target_transform,
    feature_stats,
    safe_max_memory_mb,
)
from app.vjepa.utils import init_video_mamba_model, init_video_model
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.row_tube import MaskCollator as RowTubeMaskCollator
from src.masks.utils import apply_masks


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run a synthetic one-step diagnostic smoke test.')
    parser.add_argument('--model', default='vit_tiny',
                        choices=['vit_tiny', 'videomamba_tiny'])
    parser.add_argument('--predictor', default='attention',
                        choices=['attention', 'mamba'])
    parser.add_argument('--mask-type', default='multiblock3d',
                        choices=['multiblock3d', 'random_tube', 'row_tube'])
    parser.add_argument('--target-transform', default='layer_norm',
                        choices=['raw', 'layer_norm', 'l2_norm'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-frames', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--tubelet-size', type=int, default=2)
    parser.add_argument('--pred-depth', type=int, default=2)
    parser.add_argument('--pred-embed-dim', type=int, default=96)
    parser.add_argument('--pred-head-dim', type=int, default=96)
    parser.add_argument('--output', default='diagnostic_runs/smoke_result.json')
    parser.add_argument('--allow-cpu-videomamba', action='store_true')
    return parser.parse_args()


def _device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _mask_config(mask_type):
    if mask_type == 'multiblock3d':
        return [{
            'aspect_ratio': [0.75, 1.5],
            'num_blocks': 2,
            'spatial_scale': [0.25, 0.25],
            'temporal_scale': [1.0, 1.0],
            'max_temporal_keep': 1.0,
            'max_keep': None,
        }]
    if mask_type == 'row_tube':
        return [{'ratio': 0.5, 'shared_rows_across_batch': False}]
    return [{'ratio': 0.5}]


def _make_masks(args):
    cfgs_mask = _mask_config(args.mask_type)
    collator_cls = {
        'multiblock3d': MB3DMaskCollator,
        'random_tube': TubeMaskCollator,
        'row_tube': RowTubeMaskCollator,
    }[args.mask_type]
    collator = collator_cls(
        cfgs_mask=cfgs_mask,
        crop_size=args.crop_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size)
    batch = [torch.zeros(1) for _ in range(args.batch_size)]
    _, masks_enc, masks_pred = collator(batch)
    return masks_enc, masks_pred


def _init_models(args, device):
    if args.model == 'vit_tiny':
        return init_video_model(
            device=device,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            model_name='vit_tiny',
            crop_size=args.crop_size,
            pred_depth=args.pred_depth,
            pred_embed_dim=args.pred_embed_dim,
            uniform_power=True,
            use_mask_tokens=True,
            num_mask_tokens=1,
            zero_init_mask_tokens=True,
            use_sdpa=False)

    return init_video_mamba_model(
        device=device,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        model_name='videomamba_tiny',
        use_vit_pred=args.predictor == 'attention',
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_embed_dim=args.pred_embed_dim,
        pred_head_dim=args.pred_head_dim,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=1,
        zero_init_mask_tokens=True,
        use_sdpa=False)


def _checkpoint_roundtrip(path, encoder, predictor):
    payload = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'epoch': 0,
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location='cpu')
    return sorted(loaded.keys())


def _write_result(output_path, result):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    args = _parse_args()
    device = _device(args.device)
    if args.model == 'videomamba_tiny' and device.type == 'cpu' and not args.allow_cpu_videomamba:
        result = {
            'ok': False,
            'skipped': True,
            'reason': 'VideoMamba smoke requires CUDA/ROCm unless --allow-cpu-videomamba is set.',
            'device': str(device),
        }
        _write_result(args.output, result)
        return

    torch.manual_seed(0)
    started = time.time()
    try:
        encoder, predictor = _init_models(args, device)
        target_encoder = copy.deepcopy(encoder).to(device)
        for p in target_encoder.parameters():
            p.requires_grad = False

        clips = torch.randn(
            args.batch_size,
            3,
            args.num_frames,
            args.crop_size,
            args.crop_size,
            device=device)
        masks_enc, masks_pred = _make_masks(args)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=1e-4)

        with torch.no_grad():
            h_full = target_encoder(clips)
            h_full = apply_target_transform(h_full, args.target_transform)
            h = apply_masks(h_full, masks_pred, concat=False)
        z_context = encoder(clips, masks_enc)
        z = predictor(z_context, h, masks_enc, masks_pred)
        loss = sum(torch.mean(torch.abs(zi - hi)) for zi, hi in zip(z, h)) / len(z)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        target_var, target_abs = feature_stats(h)
        pred_var, pred_abs = feature_stats(z)

        for p in encoder.parameters():
            p.requires_grad = False
        with torch.no_grad():
            frozen_tokens = encoder(clips)
            pooled = frozen_tokens.mean(dim=1)
        probe = nn.Linear(pooled.size(-1), 4).to(device)
        labels = torch.arange(args.batch_size, device=device) % 4
        probe_loss = F.cross_entropy(probe(pooled.detach()), labels)
        probe_loss.backward()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_keys = _checkpoint_roundtrip(
                os.path.join(tmpdir, 'smoke.pth.tar'),
                encoder,
                predictor)
    except Exception as exc:
        result = {
            'ok': False,
            'skipped': False,
            'model': args.model,
            'predictor': args.predictor,
            'mask_type': args.mask_type,
            'target_transform': args.target_transform,
            'device': str(device),
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
            'wall_ms': (time.time() - started) * 1000.0,
        }
        _write_result(args.output, result)
        return

    result = {
        'ok': True,
        'model': args.model,
        'predictor': args.predictor,
        'mask_type': args.mask_type,
        'target_transform': args.target_transform,
        'device': str(device),
        'loss': float(loss.detach().cpu()),
        'probe_loss': float(probe_loss.detach().cpu()),
        'target_var': target_var,
        'pred_var': pred_var,
        'target_abs_mean': target_abs,
        'pred_abs_mean': pred_abs,
        'memory_mb': safe_max_memory_mb(),
        'wall_ms': (time.time() - started) * 1000.0,
        'checkpoint_keys': checkpoint_keys,
    }

    _write_result(args.output, result)


if __name__ == '__main__':
    main()
