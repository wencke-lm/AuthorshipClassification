# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik

# 21/07/2020
# Python 3.7.3
# Windows 8
"""Exceptions related decorators and classes."""

import functools
import logging

class ScarceDataError(Exception):
    pass

class CatalogError(Exception):
    pass

def log_exception(logger):
    """
    A decorator that takes note of all exceptions thrown by
    the decorated function and logs the functioncall
    including positional and keyword arguments together with
    the traceback.
    
    Args:
        logger(logging.Logger)
    """
    def _log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pos_args = [repr(a) for a in args]
            key_args = [f"{n}={a!r}" for n,a in kwargs.items()]
            msg = f"calling {func.__name__}({', '.join(pos_args + key_args)})..."
            try:
                value = func(*args, **kwargs)
            except:
                logger.error(msg, exc_info=True)
                raise
            else:
                return value
        return wrapper
    return _log
