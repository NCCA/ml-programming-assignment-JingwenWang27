import torch
import pytest

def test_pytorch_installation():
    """
    冒烟测试：检查 PyTorch 是否能正常工作。
    """
    # 1. 测试张量创建
    x = torch.rand(5, 3)
    assert x.shape == (5, 3)
    print(f"PyTorch Version: {torch.__version__}")

def test_gpu_availability():
    """
    检查 GPU 是否可用。
    """
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("Apple Metal (MPS) is available.")
    else:
        print("CUDA/MPS not available, running on CPU.")
    assert True
