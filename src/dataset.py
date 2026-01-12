"""
Dataset Module - PBR Basecolor Dataset with Multi-Lighting Support

Loads pairs of raw rendered images and corresponding basecolor textures

"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class PBRBasecolorDataset(Dataset):
    """
    PBR Basecolor Dataset supporting multiple lighting conditions
    
    Args:
        raw_dirs: List of directories containing raw rendered images
                  
        basecolor_dir: Directory containing basecolor texture images
                      
        image_size: Target image size for resizing (default: 256)
                   
    """
    
    def __init__(
        self,
        raw_dirs: list,
        basecolor_dir: str,
        image_size: int = 256,
    ):
        self.basecolor_dir = Path(basecolor_dir)
        self.image_size = image_size
        
        # =============================================================================
        # Collect raw images from multiple lighting directories
        # =============================================================================
        self.raw_files = []
        for raw_dir in raw_dirs:
            raw_path = Path(raw_dir)
            files = sorted(raw_path.glob("*.jpg")) + sorted(raw_path.glob("*.png"))
            files += sorted(raw_path.glob("*.JPG")) + sorted(raw_path.glob("*.PNG"))
            self.raw_files.extend(files)
        
        if len(self.raw_files) == 0:
            raise ValueError(f"No images found in specified directories!")
        
        # =============================================================================
        # Match corresponding basecolor images for each raw image
        # =============================================================================
        self.basecolor_files = []
        for raw_file in self.raw_files:
            basecolor_name = self._get_basecolor_name(raw_file.name)
            basecolor_file = self.basecolor_dir / basecolor_name
            
            if not basecolor_file.exists():
                raise ValueError(f"Basecolor image not found: {basecolor_file}")
            
            self.basecolor_files.append(basecolor_file)
    
    def _get_basecolor_name(self, raw_filename: str) -> str:
        """
        Convert raw filename to basecolor filename
    
        Example: PavingStones1_01_Raw_L1.png → PavingStones1_01_Color.png
        
        """
        import re
        basecolor_name = re.sub(r'_Raw_L\d+', '_Color', raw_filename)
        return basecolor_name
    
    def __len__(self):
        """Return total number of image pairs """
        return len(self.raw_files)
    
    def __getitem__(self, idx):
        """
        Load and preprocess an image pair
                
        Returns:
            raw_tensor: Raw image tensor [3, H, W], range [0, 1]
                      
            basecolor_tensor: Basecolor image tensor [3, H, W], range [0, 1]
                            
        """
        # =============================================================================
        # Load images from disk
        # =============================================================================
        raw_img = Image.open(self.raw_files[idx]).convert("RGB")
        basecolor_img = Image.open(self.basecolor_files[idx]).convert("RGB")
        
        # =============================================================================
        # Resize images to target size for batch processing
        # =============================================================================
        raw_img = raw_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        basecolor_img = basecolor_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # =============================================================================
        # Convert PIL images to PyTorch tensors and normalize to [0, 1]
        # =============================================================================
        transform = transforms.ToTensor()
        raw_tensor = transform(raw_img)
        basecolor_tensor = transform(basecolor_img)
        
        return raw_tensor, basecolor_tensor


# =============================================================================
# Module testing code - verify dataset functionality
# =============================================================================
if __name__ == "__main__":
    dataset = PBRBasecolorDataset(
        raw_dirs=[
            "data/raw_light1",
            "data/raw_light2",
            "data/raw_light3",
            "data/raw_light4",
            "data/raw_light5",
            "data/raw_light6",
        ],
        basecolor_dir="data/basecolor",
        image_size=256,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(3):
        raw, basecolor = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Raw shape: {raw.shape}")
        print(f"  Basecolor shape: {basecolor.shape}")
        print(f"  Raw range: [{raw.min():.3f}, {raw.max():.3f}]")
        print(f"  Basecolor range: [{basecolor.min():.3f}, {basecolor.max():.3f}]")

