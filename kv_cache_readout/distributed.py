"""One frozen model per rank; only shared U gradients are reduced."""
import os
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, cfg):
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.device = torch.device(f'cuda:{self.local_rank}' if self.world > 1 else cfg['device'])
        if self.world > 1:
            torch.cuda.set_device(self.device)
            dist.init_process_group('nccl')

    def objects(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def sum(self, tensor):
        if self.world > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def any(self, flag):
        return bool(self.sum(torch.tensor(int(bool(flag)), device=self.device)).item())

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
