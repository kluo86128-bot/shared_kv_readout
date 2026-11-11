"""Explicit opt-in real-engine diagnostic; never called by training/evaluation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kv_cache_readout.launch import main
if __name__=='__main__':main('check-alignment')
