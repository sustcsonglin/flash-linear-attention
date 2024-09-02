# -*- coding: utf-8 -*-

import functools

import torch
import subprocess
import re
from functools import lru_cache
from packaging import version


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def require_version(version, hint):
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


@lru_cache(maxsize=None)
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'

    try:
        if version.parse(torch.__version__) >= version.parse('2.4'):
            if torch.xpu.is_available():
                return 'xpu'
        else:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return 'xpu'
    except ImportError:
        pass

    try:
        import torch_musa
        if torch.musa.is_available():
            return 'musa'
    except ImportError:
        pass

    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    return 'cpu'


@lru_cache(maxsize=None)
def check_compute_capacity(device):
    if device == 'cuda':
        if torch.cuda.is_available():
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True)
                compute_cap = nvidia_smi.decode('utf-8').strip()
                compute_cap_major = int(compute_cap.split('.')[0])
                return compute_cap_major >= 8
            except BaseException:
                return False
        else:
            return False

    elif device == 'xpu':
        try:
            clinfo_output = subprocess.check_output("clinfo | grep 'Max size for global variable'", shell=True)
            clinfo_output = clinfo_output.decode('utf-8').strip()
            sizes = re.findall(r'(\d+) \((\d+)KiB\)', clinfo_output)
            for size in sizes:
                if int(size[1]) > 128:
                    return True
            return False
        except BaseException:
            return False

    elif device == 'musa':
        return False

    elif device == 'npu':
        return False

    else:
        return False

device = get_available_device()
device_capacity = check_compute_capacity(device)


if version.parse(torch.__version__) >= version.parse('2.4'):
    from torch.amp import custom_fwd, custom_bwd

    def autocast_custom_fwd(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return custom_fwd(device_type=device)(args[0])
        kwargs.setdefault('device_type', device)
        return custom_fwd(**kwargs)

    def autocast_custom_bwd(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return custom_bwd(device_type=device)(args[0])
        kwargs.setdefault('device_type', device)
        return custom_bwd(**kwargs)

else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd
