"""Explicit process-local serving fixes, outside the training implementation."""

from pathlib import Path
import hashlib

POLICY = 'lmcache042_authoritative_source_positions_v1'


def identity():
    return {'policy': POLICY, 'files': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))}}
