"""Tiện ích chung: chọn device, đặt seed, trung bình trượt."""
import random

import numpy as np
import torch


def get_device(prefer_cuda=True):
    """Trả về torch.device, bật TF32 và cuDNN benchmark nếu có GPU."""
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True              # chọn kernel conv nhanh nhất cho kích thước cố định
        torch.backends.cuda.matmul.allow_tf32 = True       # TF32 trên GPU Ampere trở lên
        torch.backends.cudnn.allow_tf32 = True
    return device


def describe_device(device):
    return f"{device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x
