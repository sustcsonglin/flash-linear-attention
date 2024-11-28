# -*- coding: utf-8 -*-

import functools

import torch
from packaging import version


def contiguous(fn):
    """
    Make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    return wrapper


if version.parse(torch.__version__) >= version.parse("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd
