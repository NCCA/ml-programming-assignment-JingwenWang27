from src.model import SimpleUNet
import pytest
import torch

def test_model_architecture():
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64)
    assert model is not None, "Model should be created successfully"


def test_forward_pass():
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64)
    test_input = torch.randn(2, 3, 256, 256)
    
    output = model(test_input)
    expected_shape = (2, 3, 256, 256)
    
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"


def test_parameter_count():
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64)
    param_count = sum(p.numel() for p in model.parameters())
    
    min_params = 30_000_000
    max_params = 35_000_000
    
    assert min_params <= param_count <= max_params, \
        f"Parameter count {param_count:,} outside range [{min_params:,}, {max_params:,}]"





