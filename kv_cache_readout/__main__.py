import sys
from .launch import main
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in ('train','evaluate','check-alignment'):
        raise SystemExit('Use: python -m kv_cache_readout {train|evaluate|check-alignment} --help; pause: python scripts/pause.py')
    action=sys.argv.pop(1)
    main(action)
