import cv2
import os

def load_image(image_path):
    """
    读取图片路径，返回 numpy 数组。
    如果文件不存在，抛出 FileNotFoundError。
    """
    # 1. 检查路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 2. 读取图片
    img = cv2.imread(image_path)
    
    # 3. OpenCV 读取失败（比如文件损坏）会返回 None，我们也把它当做文件错误处理
    if img is None:
        raise FileNotFoundError(f"Failed to load image (format error?): {image_path}")
        
    return img
