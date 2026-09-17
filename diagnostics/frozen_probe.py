import argparse
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from app.vjepa.utils import init_video_mamba_model, init_video_model
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run a tiny frozen linear-probe overfit diagnostic.')
    parser.add_argument('--config', required=True,
                        help='Pretraining config used to instantiate the encoder.')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint containing encoder/target_encoder weights.')
    parser.add_argument('--manifest', default=None,
                        help='Optional manifest override. Defaults to config data.datasets[0].')
    parser.add_argument('--eval-manifest', default=None,
                        help='Optional held-out manifest evaluated after fitting the probe.')
    parser.add_argument('--checkpoint-key', default='target_encoder',
                        choices=['target_encoder', 'encoder'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', required=True,
                        help='JSON summary path.')
    parser.add_argument('--csv-output', default=None,
                        help='Optional per-epoch CSV path. Defaults next to --output.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--probe-epochs', type=int, default=40)
    parser.add_argument('--probe-lr', type=float, default=1e-2)
    parser.add_argument('--probe-weight-decay', type=float, default=0.0)
    parser.add_argument('--min-label-count', type=int, default=2,
                        help='Filter to labels appearing at least this many times.')
    parser.add_argument('--top-k-labels', type=int, default=0,
                        help='If positive, keep only the most frequent K labels after min-count filtering.')
    parser.add_argument('--seed', type=int, default=234)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def _device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load_params(config_path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _init_encoder(params, device):
    data = params['data']
    model = params['model']
    mask = params.get('mask', [{}])
    common = {
        'device': device,
        'patch_size': data.get('patch_size', 16),
        'num_frames': data.get('num_frames', 4),
        'tubelet_size': data.get('tubelet_size', 2),
        'model_name': model['model_name'],
        'crop_size': data.get('crop_size', 64),
        'pred_depth': model.get('pred_depth', 2),
        'pred_embed_dim': model.get('pred_embed_dim', 96),
        'uniform_power': model.get('uniform_power', True),
        'use_mask_tokens': model.get('use_mask_tokens', True),
        'num_mask_tokens': len(mask),
        'zero_init_mask_tokens': model.get('zero_init_mask_tokens', True),
        'use_sdpa': params.get('meta', {}).get('use_sdpa', False),
    }
    if model['model_name'].startswith('videomamba'):
        encoder, predictor = init_video_mamba_model(
            **common,
            use_vit_pred=model.get('use_vit_pred', False),
            pred_head_dim=model.get('pred_head_dim', 96),
        )
    else:
        encoder, predictor = init_video_model(**common)
    del predictor
    return encoder


def _load_encoder_weights(encoder, checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint_key not in checkpoint:
        raise KeyError(
            f'{checkpoint_key!r} not in checkpoint; keys={sorted(checkpoint.keys())}')
    msg = encoder.load_state_dict(checkpoint[checkpoint_key])
    return {
        'epoch': int(checkpoint.get('epoch', -1)),
        'missing_keys': list(msg.missing_keys),
        'unexpected_keys': list(msg.unexpected_keys),
    }


def _read_label_counts(manifest):
    labels = []
    with open(manifest, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter=' ')
        for row in reader:
            if row:
                labels.append(str(row[1]))
    return Counter(labels)


def _select_indices(dataset, min_label_count, top_k_labels):
    counts = Counter(str(label) for label in dataset.labels)
    kept = [label for label, count in counts.items() if count >= min_label_count]
    kept = sorted(kept, key=lambda label: (-counts[label], label))
    if top_k_labels > 0:
        kept = kept[:top_k_labels]
    kept = set(kept)
    indices = [
        index for index, label in enumerate(dataset.labels)
        if str(label) in kept
    ]
    label_map = {label: idx for idx, label in enumerate(sorted(kept))}
    return indices, label_map, counts


def _unwrap_clip(item):
    while isinstance(item, (list, tuple)):
        item = item[0]
    return item


def _make_collate(label_map):
    def collate(batch):
        clips = []
        labels = []
        for clip_list, label, _clip_indices in batch:
            clips.append(_unwrap_clip(clip_list))
            labels.append(label_map[str(label)])
        return torch.stack(clips), torch.tensor(labels, dtype=torch.long)
    return collate


def _select_indices_for_label_map(dataset, label_map):
    return [
        index for index, label in enumerate(dataset.labels)
        if str(label) in label_map
    ]


def _extract_features(encoder, loader, device):
    features = []
    labels = []
    started = time.time()
    with torch.no_grad():
        for clips, batch_labels in loader:
            clips = clips.to(device, non_blocking=True)
            tokens = encoder(clips)
            pooled = tokens.mean(dim=1)
            features.append(pooled.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    elapsed = time.time() - started
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels, elapsed


def _train_probe(features, labels, args, device, csv_path):
    torch.manual_seed(args.seed)
    features = features.to(device)
    labels = labels.to(device)
    classifier = nn.Linear(features.shape[1], int(labels.max().item()) + 1).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'train_acc'])
        for epoch in range(1, args.probe_epochs + 1):
            order = torch.randperm(features.shape[0], device=device)
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for start in range(0, features.shape[0], args.batch_size):
                batch_idx = order[start:start + args.batch_size]
                logits = classifier(features[batch_idx])
                loss = F.cross_entropy(logits, labels[batch_idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = int(batch_idx.numel())
                total_loss += float(loss.detach().cpu()) * batch_size
                total_correct += int((logits.argmax(dim=1) == labels[batch_idx]).sum().item())
                total_seen += batch_size

            row = {
                'epoch': epoch,
                'train_loss': total_loss / total_seen,
                'train_acc': total_correct / total_seen,
            }
            rows.append(row)
            writer.writerow([
                row['epoch'],
                f'{row["train_loss"]:.6f}',
                f'{row["train_acc"]:.6f}',
            ])
    return classifier, rows, time.time() - started


def _eval_probe(classifier, features, labels, device):
    features = features.to(device)
    labels = labels.to(device)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
    classifier.train()
    return float(loss.detach().cpu()), float(acc.detach().cpu())


def main():
    args = _parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    params = _load_params(args.config)
    manifest = args.manifest or params['data']['datasets'][0]
    csv_output = (
        Path(args.csv_output)
        if args.csv_output
        else Path(args.output).with_suffix('.csv')
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    encoder = _init_encoder(params, device)
    load_info = _load_encoder_weights(encoder, args.checkpoint, args.checkpoint_key)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    data = params['data']
    transform = make_transforms(
        training=False,
        crop_size=data.get('crop_size', 64),
        num_views_per_clip=1,
    )
    dataset = VideoDataset(
        data_paths=[manifest],
        frames_per_clip=data.get('num_frames', 4),
        frame_step=data.get('sampling_rate', 2),
        num_clips=1,
        random_clip_sampling=False,
        allow_clip_overlap=True,
        filter_short_videos=data.get('filter_short_videos', True),
        transform=transform,
    )
    indices, label_map, raw_counts = _select_indices(
        dataset,
        min_label_count=args.min_label_count,
        top_k_labels=args.top_k_labels,
    )
    if not indices:
        raise ValueError('No probe samples left after label filtering')

    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_make_collate(label_map),
        drop_last=False,
    )

    features, labels, feature_seconds = _extract_features(encoder, loader, device)
    classifier, rows, probe_seconds = _train_probe(features, labels, args, device, csv_output)
    eval_result = None
    if args.eval_manifest is not None:
        eval_dataset = VideoDataset(
            data_paths=[args.eval_manifest],
            frames_per_clip=data.get('num_frames', 4),
            frame_step=data.get('sampling_rate', 2),
            num_clips=1,
            random_clip_sampling=False,
            allow_clip_overlap=True,
            filter_short_videos=data.get('filter_short_videos', True),
            transform=transform,
        )
        eval_indices = _select_indices_for_label_map(eval_dataset, label_map)
        if not eval_indices:
            raise ValueError('No eval samples match the training label map')
        eval_subset = torch.utils.data.Subset(eval_dataset, eval_indices)
        eval_loader = torch.utils.data.DataLoader(
            eval_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_make_collate(label_map),
            drop_last=False,
        )
        eval_features, eval_labels, eval_feature_seconds = _extract_features(
            encoder, eval_loader, device)
        eval_loss, eval_acc = _eval_probe(classifier, eval_features, eval_labels, device)
        eval_result = {
            'eval_manifest': args.eval_manifest,
            'eval_raw_clips': len(eval_dataset),
            'eval_clips': int(eval_features.shape[0]),
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'eval_feature_seconds': eval_feature_seconds,
        }
    final = rows[-1]
    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'checkpoint_key': args.checkpoint_key,
        'checkpoint_epoch': load_info['epoch'],
        'load_missing_keys': load_info['missing_keys'],
        'load_unexpected_keys': load_info['unexpected_keys'],
        'manifest': manifest,
        'eval_manifest': args.eval_manifest,
        'device': str(device),
        'seed': args.seed,
        'min_label_count': args.min_label_count,
        'top_k_labels': args.top_k_labels,
        'raw_clips': len(dataset),
        'raw_classes': len(raw_counts),
        'probe_clips': int(features.shape[0]),
        'probe_classes': int(labels.max().item()) + 1,
        'feature_dim': int(features.shape[1]),
        'feature_var': float(features.var(dim=0, unbiased=False).mean().item()),
        'feature_abs_mean': float(features.abs().mean().item()),
        'probe_epochs': args.probe_epochs,
        'probe_lr': args.probe_lr,
        'probe_weight_decay': args.probe_weight_decay,
        'final_train_loss': final['train_loss'],
        'final_train_acc': final['train_acc'],
        'best_train_acc': max(row['train_acc'] for row in rows),
        'csv': str(csv_output),
        'feature_seconds': feature_seconds,
        'probe_seconds': probe_seconds,
        'total_seconds': time.time() - started,
        'note': (
            'Without --eval-manifest this is a frozen linear-probe overfit '
            'diagnostic on a tiny, label-sparse local subset; with '
            '--eval-manifest it is a held-out subset diagnostic.'
        ),
    }
    if eval_result is not None:
        result.update(eval_result)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
