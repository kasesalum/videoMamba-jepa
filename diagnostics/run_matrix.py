import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and optionally run the phase-1 diagnostic matrix.')
    parser.add_argument(
        '--matrix',
        default='configs/diagnostics/phase1_matrix.yaml',
        help='YAML matrix specification.')
    parser.add_argument(
        '--write-configs',
        action='store_true',
        help='Write generated per-run configs.')
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run generated commands after passing the budget gate.')
    parser.add_argument(
        '--measured-ms',
        type=float,
        default=None,
        help='Override estimated wall ms per microstep with smoke/benchmark data.')
    parser.add_argument(
        '--output',
        default='diagnostic_runs/phase1_plan.json',
        help='Where to write the generated plan JSON.')
    return parser.parse_args()


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _set_path(data, dotted_path, value):
    parts = dotted_path.split('.')
    cursor = data
    for part in parts[:-1]:
        if part not in cursor or cursor[part] is None:
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _deep_update(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _apply_variant(base, variant, output_root):
    cfg = copy.deepcopy(base)
    run_name = variant['name']
    _set_path(cfg, 'logging.folder', str(output_root / run_name))
    _set_path(cfg, 'logging.write_tag', run_name)
    for key, value in variant.get('patch', {}).items():
        _set_path(cfg, key, value)
    if 'patch_tree' in variant:
        _deep_update(cfg, variant['patch_tree'])
    return cfg


def _trainer_name(cfg):
    trainer = cfg.get('trainer')
    if trainer:
        return trainer
    model_name = cfg.get('model', {}).get('model_name', '')
    return 'train_videomamba' if model_name.startswith('videomamba') else 'train'


def _microsteps(cfg):
    opt = cfg.get('optimization', {})
    epochs = int(opt.get('epochs', 1))
    ipe = int(opt.get('ipe', 1))
    accum = int(opt.get('accumulation_steps', 1))
    if _trainer_name(cfg) != 'train_videomamba':
        accum = 1
    return epochs * ipe * accum


def _estimate_lumi_gpu_hours(cfg, budget, wall_ms_per_microstep):
    nodes = int(cfg.get('nodes', 1))
    modules_per_node = float(budget.get('lumi_gpu_modules_per_node', 4))
    wall_hours = _microsteps(cfg) * wall_ms_per_microstep / 1000.0 / 3600.0
    return nodes * modules_per_node * wall_hours


def _command(cfg_path, run_cfg):
    launch = run_cfg.get('launch', 'local')
    if launch == 'slurm':
        folder = run_cfg.get('slurm_folder', 'diagnostic_runs/slurm')
        partition = run_cfg.get('partition', '$slurm_partition')
        return [
            'python', '-m', 'app.main_distributed',
            '--fname', str(cfg_path),
            '--folder', folder,
            '--partition', partition,
        ]
    devices = run_cfg.get('devices', ['cuda:0'])
    return [
        'python', '-m', 'app.main',
        '--fname', str(cfg_path),
        '--devices',
        *devices,
    ]


def main():
    args = _parse_args()
    matrix = _load_yaml(args.matrix)
    base = _load_yaml(matrix['base_config'])
    output_root = Path(matrix.get('output_root', 'diagnostic_runs/phase1'))
    config_root = Path(matrix.get('config_root', 'diagnostic_runs/configs'))
    budget = matrix.get('budget', {})

    wall_ms = args.measured_ms
    if wall_ms is None:
        wall_ms = float(budget.get('estimated_wall_ms_per_microstep', 2000.0))

    eur_cap = float(budget.get('eur_cap', 1000.0))
    price = float(budget.get('price_eur_per_lumi_gpu_hour', 0.35))
    setup_fraction = float(budget.get('setup_fraction', 0.10))
    reserve_fraction = float(budget.get('reserve_fraction', 0.20))
    training_eur_cap = eur_cap * max(0.0, 1.0 - setup_fraction - reserve_fraction)
    training_gpu_hour_cap = training_eur_cap / price

    plan = {
        'matrix': args.matrix,
        'wall_ms_per_microstep': wall_ms,
        'eur_cap': eur_cap,
        'training_eur_cap': training_eur_cap,
        'training_lumi_gpu_hour_cap': training_gpu_hour_cap,
        'runs': [],
    }

    total_gpu_hours = 0.0
    for variant in matrix['variants']:
        cfg = _apply_variant(base, variant, output_root)
        cfg_path = config_root / f"{variant['name']}.yaml"
        lumi_gpu_hours = _estimate_lumi_gpu_hours(cfg, budget, wall_ms)
        cost_eur = lumi_gpu_hours * price
        total_gpu_hours += lumi_gpu_hours
        run = {
            'name': variant['name'],
            'config': str(cfg_path),
            'trainer': _trainer_name(cfg),
            'model': cfg.get('model', {}).get('model_name'),
            'mask_type': cfg.get('data', {}).get('mask_type'),
            'target_transform': cfg.get('model', {}).get('target_transform', 'layer_norm'),
            'microsteps': _microsteps(cfg),
            'estimated_lumi_gpu_hours': lumi_gpu_hours,
            'estimated_cost_eur': cost_eur,
            'command': _command(cfg_path, matrix.get('run', {})),
        }
        plan['runs'].append(run)

        if args.write_configs or args.run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(yaml.dump(cfg, sort_keys=False))

    plan['total_estimated_lumi_gpu_hours'] = total_gpu_hours
    plan['total_estimated_cost_eur'] = total_gpu_hours * price
    plan['budget_ok'] = total_gpu_hours <= training_gpu_hour_cap

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))

    if not plan['budget_ok']:
        raise SystemExit(
            'Budget gate failed: estimated training matrix cost exceeds '
            f'{training_eur_cap:.2f} EUR after setup/reserve.')

    if args.run:
        for run in plan['runs']:
            Path(run['config']).parent.mkdir(parents=True, exist_ok=True)
            print('RUN', ' '.join(run['command']))
            subprocess.run(run['command'], check=True)


if __name__ == '__main__':
    main()
