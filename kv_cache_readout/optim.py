"""Original three-segment update and warmup/cosine rules."""
import math
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def parameters_for(initial, cfg):
    sizes = [cfg['prompt_partition'][key] for key in ('other_length', 'focal_length', 'keep_length')]
    return [torch.nn.Parameter(part.clone()) for part in initial.split(sizes)]


def assign_gradients(parameters, gradient, active):
    for parameter, part, enabled in zip(parameters, gradient.split([len(p) for p in parameters]), active):
        parameter.grad = part.clone() if enabled else None


def schedule_factor(step, total, fraction):
    warmup = max(1, math.ceil(total * fraction)) if fraction else 0
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))


def make_scheduler(optimizer, total, fraction):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(step, total, fraction))
