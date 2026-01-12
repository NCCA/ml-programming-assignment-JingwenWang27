"""
Unit Tests for SimpleUNet Model
"""
import torch
import pytest

from src.model import SimpleUNet


# =============================================================================
# Test 1: Basic input/output functionality

# =============================================================================
def test_model_basic():
    """Test: Model runs correctly with proper output shape"""
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    
    assert y.shape == torch.Size([1, 3, 256, 256])


# =============================================================================
# Test 2: Batch processing

# =============================================================================
def test_model_batch():
    """Test: Model can process multiple images in batch"""
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    # Test with batch_size=4
    x = torch.randn(4, 3, 256, 256)
    y = model(x)
    
    assert y.shape == torch.Size([4, 3, 256, 256])


# =============================================================================
# Test 3: Different input sizes
# =============================================================================
def test_model_different_sizes():
    """Test: Model handles different image sizes"""
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    # Test 128x128
    x1 = torch.randn(1, 3, 128, 128)
    y1 = model(x1)
    assert y1.shape == torch.Size([1, 3, 128, 128])
    
    # Test 512x512
    x2 = torch.randn(1, 3, 512, 512)
    y2 = model(x2)
    assert y2.shape == torch.Size([1, 3, 512, 512])


# =============================================================================
# Test 4: Gradient backpropagation
# =============================================================================
def test_model_backward():
    """Test: Model supports training with proper gradient flow"""
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    y = model(x)
    
    # Compute loss and backpropagate
    loss = y.sum()
    loss.backward()
    
    # Verify gradients exist
    assert x.grad is not None


# =============================================================================
# Test 5: Model save and load
# =============================================================================
def test_model_save_load(tmp_path):
    """Test: Model can be saved and loaded correctly"""
    model1 = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    # Save model
    save_path = tmp_path / "model.pth"
    torch.save(model1.state_dict(), save_path)
    
    # Load into new model
    model2 = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    model2.load_state_dict(torch.load(save_path))
    
    # Verify loaded model works correctly
    x = torch.randn(1, 3, 256, 256)
    y = model2(x)
    assert y.shape == torch.Size([1, 3, 256, 256])


# =============================================================================
# Test 6: Output validity (no NaN/Inf)
# =============================================================================
def test_model_output_valid():
    """Test: Model output contains no NaN or Inf values"""
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=32)
    
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    
    # Check for no NaN or Inf values
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
