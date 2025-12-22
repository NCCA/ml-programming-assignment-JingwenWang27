import pytest
import os
import numpy as np
import cv2
# 下面这行现在会报错，因为我们还没写 src 里的代码，这是正常的！
from src.data_loader import load_image

def test_load_image_file_not_found():
    """测试 1: 读取不存在的文件应该报错"""
    fake_path = "non_existent_image.jpg"
    with pytest.raises(FileNotFoundError):
        load_image(fake_path)

def test_load_image_success():
    """测试 2: 读取正常的图片应该成功"""
    # 造一张假图片
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_path = "temp_test_image.jpg"
    cv2.imwrite(dummy_path, dummy_img)

    try:
        img = load_image(dummy_path)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)
    finally:
        # 清理垃圾
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
