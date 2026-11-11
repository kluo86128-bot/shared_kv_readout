"""Portable four/single/multi-GPU training entry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kv_cache_readout.launch import main
if __name__=='__main__':main('train')
