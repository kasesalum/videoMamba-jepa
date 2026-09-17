import torch
import torch.nn.functional as F


def apply_target_transform(targets, mode):
    mode = (mode or 'layer_norm').lower()
    if mode in {'layer_norm', 'normalized', 'normalize'}:
        return F.layer_norm(targets, (targets.size(-1),))
    if mode in {'l2_norm', 'unit_norm'}:
        return F.normalize(targets, dim=-1)
    if mode in {'raw', 'none', 'identity'}:
        return targets
    raise ValueError(
        f'Unsupported target_transform={mode!r}. '
        'Use one of: raw, layer_norm, l2_norm.')


def feature_stats(features):
    if not isinstance(features, (list, tuple)):
        features = [features]
    with torch.no_grad():
        flat = torch.cat(
            [f.detach().reshape(-1, f.size(-1)).float() for f in features],
            dim=0)
        return float(flat.var(dim=0).mean()), float(flat.abs().mean())


def gradient_global_norm(parameters):
    norms = [
        torch.norm(p.grad.detach().float())
        for p in parameters
        if p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms)))


def safe_max_memory_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024.0**2


def clips_per_second(local_batch, world_size, wall_time_ms):
    if wall_time_ms <= 0:
        return 0.0
    return float(local_batch * world_size) / (wall_time_ms / 1000.0)


def device_hours(world_size, wall_time_ms):
    return float(world_size) * (wall_time_ms / 1000.0) / 3600.0
