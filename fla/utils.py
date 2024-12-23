# -*- coding: utf-8 -*-

import functools
from typing import Any, Callable, Optional, Tuple

import torch
from packaging import version


def contiguous(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.

    NOTE:
        This cache uses tensor memory addresses as keys, so it won't detect changes to tensor contents if modified in-place.
    """
    last_key: Optional[Tuple] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_key, last_result

        # Convert all tensor arguments to a single hash key
        key = tuple(arg.data_ptr() for arg in args if torch.is_tensor(arg))
        key += tuple((k, v.data_ptr()) for k, v in kwargs.items() if torch.is_tensor(v))

        if key == last_key and last_result is not None:
            return last_result

        result = fn(*args, **kwargs)
        last_key, last_result = key, result
        return result

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


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


if version.parse(torch.__version__) >= version.parse("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd
