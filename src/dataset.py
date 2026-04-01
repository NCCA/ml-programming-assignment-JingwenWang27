from torch.utils.data import Dataset, DataLoader, random_split

class PBRBasecolorDataset(Dataset):
    """
    Dataset for PBR basecolor prediction with data augmentation support.
    """
    
    def __init__(self, raw_dirs, basecolor_dir, image_size=256, augment=False):
        self.basecolor_dir = Path(basecolor_dir)
        self.image_size = image_size
        self.augment = augment  
        
        # Collect all rendered images from 6 lighting folders
        self.raw_files = []
        for raw_dir in raw_dirs:
            raw_path = Path(raw_dir)
            files = sorted(raw_path.glob("*.jpg")) + sorted(raw_path.glob("*.png"))
            files += sorted(raw_path.glob("*.JPG")) + sorted(raw_path.glob("*.PNG"))
            self.raw_files.extend(files)
        
        # Find corresponding basecolor for each rendered image
        self.basecolor_files = []
        for raw_file in self.raw_files:
            # Filename conversion: Material_Raw_L1.png → Material_Color.png
            import re
            basecolor_name = re.sub(r'_Raw_L\d+', '_Color', raw_file.name)
            basecolor_file = self.basecolor_dir / basecolor_name
            self.basecolor_files.append(basecolor_file)
    
    def __len__(self):
        return len(self.raw_files)
    
    def __getitem__(self, idx):
        # Load images
        raw_img = Image.open(self.raw_files[idx]).convert("RGB")
        basecolor_img = Image.open(self.basecolor_files[idx]).convert("RGB")
        
        # Resize to 256x256
        raw_img = raw_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        basecolor_img = basecolor_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                basecolor_img = basecolor_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random rotation (90, 180, 270 degrees)
            if torch.rand(1) > 0.5:
                angle = torch.randint(1, 4, (1,)).item() * 90  # 90, 180, or 270
                raw_img = raw_img.rotate(angle)
                basecolor_img = basecolor_img.rotate(angle)
       
        
        # Convert to tensor, auto normalize to [0,1]
        transform = transforms.ToTensor()
        raw_tensor = transform(raw_img)
        basecolor_tensor = transform(basecolor_img)
        
        return raw_tensor, basecolor_tensor

