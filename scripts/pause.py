"""Cooperative pause: standard library only; never loads torch or a model."""
import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir');p.add_argument('--config',default='configs/train_hr.json')
    args=p.parse_args()
    config=Path(args.config);config=config if config.is_absolute() else ROOT/config
    directory=args.run_dir or json.loads(config.read_text(encoding='utf-8'))['output_directory']
    root=Path(directory);root=root if root.is_absolute() else ROOT/root
    if not root.is_dir():raise SystemExit(f'Run directory not found: {root}')
    target=root/'pause.request';temporary=target.with_name(f'.pause.{os.getpid()}.tmp')
    temporary.write_text('pause at next committed update/question wave\n',encoding='utf-8')
    os.replace(temporary,target)
    print(f'Pause requested: {target}',flush=True)

if __name__=='__main__':main()
