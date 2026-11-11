"""Stable random C1 subsets, independent of rank, batching, epoch and global RNG."""
import hashlib
import math
import random


def settings(cfg):
    return cfg.get('c1_selection',dict(method='cacheblend',seed=cfg['seed']))


def validate(value):
    if set(value)!={'method','seed'} or value['method'] not in ('cacheblend','random','epic','none'):
        raise ValueError('c1_selection requires method=cacheblend/random/epic/none and seed')
    if type(value['seed']) is not int or value['seed']<0:
        raise ValueError('C1 selection seed must be a nonnegative integer')


def random_indices(length,ratio,seed,record_id,device):
    import torch
    if length<1 or not math.isfinite(ratio) or not 0<=ratio<=1:
        raise ValueError('Invalid random C1 length or ratio')
    count=0 if ratio==0 else max(int(length*ratio),1)
    digest=hashlib.sha256(f'{seed}|{record_id}|C-recomputation-v1'.encode()).digest()
    order=list(range(length));random.Random(int.from_bytes(digest,'big')).shuffle(order)
    return torch.tensor(sorted(order[:count]),device=device,dtype=torch.long)


def check_indices(indices,length,ratio,device):
    import torch
    count=0 if ratio==0 else max(int(length*ratio),1)
    if (not isinstance(indices,torch.Tensor) or indices.dtype!=torch.long or indices.ndim!=1
            or indices.numel()!=count or indices.device!=device):
        raise ValueError('Random C1 indices must be a device-local long tensor with the exact budget')
    if count and (int(indices.min())<0 or int(indices.max())>=length
                  or not torch.equal(indices,indices.unique(sorted=True))):
        raise ValueError('Random C1 indices must be sorted, unique and in range')
