"""Stage detail logging is disabled; optimizer summaries use [TRAIN]."""


def emit(cfg,stage,**fields):
    # Keep existing call sites and numerical execution unchanged, without
    # writing stage messages to stdout/stderr or the redirected log file.
    return
