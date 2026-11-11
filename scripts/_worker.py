"""Subprocess boundaries keep engine/export/training CUDA lifetimes separate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from kv_cache_readout.config import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','alignment','evaluate'))
    p.add_argument('--config',required=True);p.add_argument('--checkpoint')
    p.add_argument('--stop-after-epochs',type=int);p.add_argument('--max-updates',type=int)
    p.add_argument('--phase');p.add_argument('--root');p.add_argument('--world',type=int,default=1)
    p.add_argument('--workload-config');p.add_argument('--validation-groups',type=int,default=4)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--split',default='test');p.add_argument('--limit',type=int)
    args=p.parse_args();cfg=load_config(args.config)
    if args.action=='train':
        from kv_cache_readout.training import train
        print(train(cfg,resume_from=args.checkpoint,
                    stop_after_epochs=args.stop_after_epochs,max_updates=args.max_updates),flush=True)
    elif args.action=='alignment':
        from kv_cache_readout.alignment import phase
        phase(args.phase,cfg,args.root,args.world,args.checkpoint,args.workload_config,args.validation_groups)
    else:
        from kv_cache_readout.offline import evaluate_worker
        evaluate_worker(cfg,args.checkpoint,args.rank,args.world,args.split,args.limit)

if __name__=='__main__':main()
