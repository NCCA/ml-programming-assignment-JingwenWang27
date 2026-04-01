"""
Inference Script - Basecolor Prediction from Rendered Images

Functionality:

- Use trained model to predict basecolor from new images

- Convert lit rendered images to basecolor textures

- Support single image, batch, and multi-lighting inference

"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.model import SimpleUNet


def load_image_as_tensor(image_path, image_size=256, device="cpu"):
    """
    Load single image and convert to tensor
    
    Args:
        image_path (str): Path to image file
    
        image_size (int): Target image size after resizing
                   
        device (str): 'cpu' or 'cuda'
               
    
    Returns:
        Tensor: shape=[1, 3, H, W], range [0, 1]
         
    """
    # Load and convert image to RGB
    img = Image.open(image_path).convert("RGB")
    
    # Resize to target size
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_np = np.array(img).astype("float32") / 255.0  # [H, W, 3]
    
    # Transpose to channel-first format
    img_np = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    return img_tensor


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: shape=[1, 3, H, W] or [3, H, W], range [0, 1]
    
    Returns:
        PIL.Image: RGB image

    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [3, H, W]
    
    # Move to CPU and convert to numpy
    img_np = tensor.detach().cpu().numpy()  # [3, H, W]
    
    # Transpose to channel-last format
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, 3]
    
    # Clip to valid range and convert to uint8
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype("uint8")
    
    # Convert to PIL Image
    img = Image.fromarray(img_np, mode="RGB")
    
    return img


def load_model(model_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        model_path (str): Path to model checkpoint

        device: torch.device object
    
    Returns:
        model: Loaded model in evaluation mode

    """
    print(f"\nLoading model...")
    
    # Check if model file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
    ).to(device)
    
    # Load checkpoint (supports two formats)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: contains epoch, loss, etc.
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Training loss: {checkpoint.get('loss', 'N/A'):.6f}")
    else:
        # Old format: only state_dict
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully")
    
    model.eval()  # Set to evaluation mode
    
    return model


def infer_single_image(model, image_path, output_path, image_size, device):
    """
    Perform inference on a single image
    
    Args:
        model: Trained model

        image_path: Input image path

        output_path: Output image path

        image_size: Image size for processing

        device: Computation device

    """
    with torch.no_grad():
        # Load image as tensor
        raw_tensor = load_image_as_tensor(
            image_path, 
            image_size=image_size,
            device=device
        )
        
        # Model prediction
        pred_tensor = model(raw_tensor)
        
        # Convert to image and save
        pred_img = tensor_to_image(pred_tensor)
        pred_img.save(output_path)


def infer_directory(model, input_dir, output_dir, image_size, device):
    """
    Perform batch inference on all images in a directory

    
    Args:
        model: Trained model

        input_dir: Input directory path

        output_dir: Output directory path

        image_size: Image size for processing

        device: Computation device

    
    Returns:
        int: Number of processed images

    """
    # Scan for input images
    input_path = Path(input_dir)
    image_files = []
    
    # Supported image formats
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"Warning: No image files found in {input_dir}")
        return 0
    
    print(f"  Found {len(image_files)} images")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Batch inference
    with torch.no_grad():
        for img_file in tqdm(image_files, desc=f"  Processing {input_path.name}"):
            # Generate output filename
            output_filename = img_file.stem + "_predicted.png"
            output_file = output_path / output_filename
            
            # Perform inference
            infer_single_image(
                model=model,
                image_path=img_file,
                output_path=output_file,
                image_size=image_size,
                device=device
            )
    
    return len(image_files)


def infer():
    """Main inference function """
    
    # =============================================================================
    # Configuration parameters

    # =============================================================================
    model_path = "checkpoints/best_model.pth"
    
    # Support multiple lighting condition directories
    input_dirs = [
        "data/raw_light1",
        "data/raw_light2",
        "data/raw_light3",
        "data/raw_light4",
        "data/raw_light5",
        "data/raw_light6",
    ]
    
    output_base_dir = "output"
    image_size = 256
    
    # =============================================================================
    # Device setup
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Starting Inference")
    print("=" * 60)
    print(f"Using device: {device}")
    
    print(f"\nPath configuration:")
    print(f"  Model path: {model_path}")
    print(f"  Number of input directories: {len(input_dirs)}")
    print(f"  Output base directory: {output_base_dir}")
    
    # =============================================================================
    # Load model
    # =============================================================================
    model = load_model(model_path, device)
    
    # =============================================================================
    # Batch inference for all lighting conditions
    # =============================================================================
    print(f"\n{'=' * 60}")
    print(f"Starting batch inference")
    print(f"{'=' * 60}\n")
    
    total_images = 0
    
    for input_dir in input_dirs:
        # Check if directory exists
        if not Path(input_dir).exists():
            print(f"Warning: Skipping non-existent directory: {input_dir}")
            continue
        
        # Generate output directory name (e.g., output/light1_predicted)
        dir_name = Path(input_dir).name  # raw_light1
        output_dir = Path(output_base_dir) / f"{dir_name}_predicted"
        
        print(f"\nProcessing directory: {input_dir}")
        print(f"  Output to: {output_dir}")
        
        # Perform inference
        num_images = infer_directory(
            model=model,
            input_dir=input_dir,
            output_dir=output_dir,
            image_size=image_size,
            device=device
        )
        
        total_images += num_images
        print(f"Completed {num_images} images")
    
    # =============================================================================
    # Completion summary
    # =============================================================================
    print(f"\n{'=' * 60}")
    print(f"Inference completed!")
    print(f"  Total processed images: {total_images}")
    print(f"  Results saved in: {output_base_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    infer()

