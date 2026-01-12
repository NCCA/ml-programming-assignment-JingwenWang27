"""
Unit Tests for PBRBasecolorDataset

"""
import os
import pytest
from PIL import Image
from pathlib import Path

from src.dataset import PBRBasecolorDataset


# =============================================================================
# Helper function: Create test images
# =============================================================================
def create_test_image(path, size=(512, 512)):
    """
    Create a simple test image

    
    Args:
        path: Output file path 
        size: Image dimensions 
    """
    img = Image.new('RGB', size, color=(100, 150, 200))
    img.save(path)


# =============================================================================
# Test 1: Empty dataset should raise error
# =============================================================================
def test_dataset_empty(tmp_path):
    """Test: Empty directories should raise RuntimeError"""
    raw_dir = tmp_path / "raw"
    basecolor_dir = tmp_path / "basecolor"
    raw_dir.mkdir()
    basecolor_dir.mkdir()
    
    # Empty directories should raise error
    with pytest.raises(RuntimeError, match="enpty"):
        dataset = PBRBasecolorDataset(
            raw_dir=str(raw_dir),
            basecolor_dir=str(basecolor_dir),
            image_size=256,
        )


# =============================================================================
# Test 2: Dataset length with images
# =============================================================================
def test_dataset_with_images(tmp_path):
    """Test: Dataset with 3 images should have length 3"""
    raw_dir = tmp_path / "raw"
    basecolor_dir = tmp_path / "basecolor"
    raw_dir.mkdir()
    basecolor_dir.mkdir()
    
    # Create 3 pairs of images
    for i in range(3):
        create_test_image(raw_dir / f"img_{i}.jpg")
        create_test_image(basecolor_dir / f"img_{i}.jpg")
    
    dataset = PBRBasecolorDataset(
        raw_dir=str(raw_dir),
        basecolor_dir=str(basecolor_dir),
        image_size=256,
    )
    
    assert len(dataset) == 3


# =============================================================================
# Test 3: Data format validation
# =============================================================================
def test_dataset_getitem(tmp_path):
    """Test: Retrieved data should have correct format"""
    raw_dir = tmp_path / "raw"
    basecolor_dir = tmp_path / "basecolor"
    raw_dir.mkdir()
    basecolor_dir.mkdir()
    
    # Create 1 pair of images
    create_test_image(raw_dir / "test.jpg")
    create_test_image(basecolor_dir / "test.jpg")
    
    dataset = PBRBasecolorDataset(
        raw_dir=str(raw_dir),
        basecolor_dir=str(basecolor_dir),
        image_size=256,
    )
    
    # Retrieve first data sample
    raw_tensor, basecolor_tensor = dataset[0]
    
    # Verify tensor shapes
    assert raw_tensor.shape == (3, 256, 256)
    assert basecolor_tensor.shape == (3, 256, 256)


# =============================================================================
# Test 4: Missing basecolor image should raise error
# =============================================================================
def test_dataset_missing_basecolor(tmp_path):
    """Test: Missing basecolor image should raise RuntimeError"""
    raw_dir = tmp_path / "raw"
    basecolor_dir = tmp_path / "basecolor"
    raw_dir.mkdir()
    basecolor_dir.mkdir()
    
    # Create only raw image, no basecolor
    create_test_image(raw_dir / "test.jpg")
    
    dataset = PBRBasecolorDataset(
        raw_dir=str(raw_dir),
        basecolor_dir=str(basecolor_dir),
        image_size=256,
    )
    
    # Should raise error when accessing data
    with pytest.raises(RuntimeError):
        _ = dataset[0]


# =============================================================================
# Test 5: Different image sizes
# =============================================================================
def test_dataset_different_sizes(tmp_path):
    """Test: Dataset should support different output sizes"""
    raw_dir = tmp_path / "raw"
    basecolor_dir = tmp_path / "basecolor"
    raw_dir.mkdir()
    basecolor_dir.mkdir()
    
    create_test_image(raw_dir / "test.jpg")
    create_test_image(basecolor_dir / "test.jpg")
    
    # Test both 128 and 512 sizes
    for size in [128, 512]:
        dataset = PBRBasecolorDataset(
            raw_dir=str(raw_dir),
            basecolor_dir=str(basecolor_dir),
            image_size=size,
        )
        
        raw_tensor, basecolor_tensor = dataset[0]
        assert raw_tensor.shape == (3, size, size)
