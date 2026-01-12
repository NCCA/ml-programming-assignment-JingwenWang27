"""

Model Module - Improved U-Net Architecture

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    """
    Improved U-Net for image-to-image translation
    
    Improvements:

    1. Increased channel capacity: base_channels = 64 (was 32)
   
    2. Deeper architecture: 4 encoder layers (was 3)
       
    3. Enhanced feature extraction capability
     
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3, 
        base_channels: int = 64
    ):
        super().__init__()

        # =============================================================================
        # Encoder: Progressively downsample and extract hierarchical features
        # =============================================================================
        self.enc1 = self._block(in_channels, base_channels)           # 3 → 64
        self.enc2 = self._block(base_channels, base_channels * 2)     # 64 → 128
        self.enc3 = self._block(base_channels * 2, base_channels * 4) # 128 → 256
        self.enc4 = self._block(base_channels * 4, base_channels * 8) # 256 → 512

        # =============================================================================
        # Bottleneck: Capture the most abstract features at lowest resolution
        # =============================================================================
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)  # 512 → 1024

        # =============================================================================
        # Decoder: Progressively upsample and refine features with skip connections
        # =============================================================================
        self.dec4 = self._block(base_channels * 16 + base_channels * 8, base_channels * 8)  # 1024+512 → 512
        self.dec3 = self._block(base_channels * 8 + base_channels * 4, base_channels * 4)   # 512+256 → 256
        self.dec2 = self._block(base_channels * 4 + base_channels * 2, base_channels * 2)   # 256+128 → 128
        self.dec1 = self._block(base_channels * 2 + base_channels, base_channels)           # 128+64 → 64

        # =============================================================================
        # Output layer: Convert features to RGB image
        # =============================================================================
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _block(self, in_c: int, out_c: int):
        """
        Double convolution block with batch normalization
        
        Structure: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU

        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Output tensor [B, 3, H, W]

        """
        
        # =============================================================================
        # Encoder path: Extract features at multiple scales
        # =============================================================================
        enc1_out = self.enc1(x)                          # [B, 64, H, W]
        enc2_out = self.enc2(F.max_pool2d(enc1_out, 2))  # [B, 128, H/2, W/2]
        enc3_out = self.enc3(F.max_pool2d(enc2_out, 2))  # [B, 256, H/4, W/4]
        enc4_out = self.enc4(F.max_pool2d(enc3_out, 2))  # [B, 512, H/8, W/8]

        # =============================================================================
        # Bottleneck: Process features at lowest resolution
        # =============================================================================
        bottleneck_out = self.bottleneck(F.max_pool2d(enc4_out, 2))  # [B, 1024, H/16, W/16]

        # =============================================================================
        # Decoder path: Upsample and combine with encoder features via skip connections
        # =============================================================================
        
        # Decoder level 4
        up4 = F.interpolate(bottleneck_out, scale_factor=2, mode="bilinear", align_corners=False)
        up4 = torch.cat([up4, enc4_out], dim=1)  # Concatenate skip connection
        dec4_out = self.dec4(up4)
        
        # Decoder level 3
        up3 = F.interpolate(dec4_out, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = torch.cat([up3, enc3_out], dim=1)
        dec3_out = self.dec3(up3)

        # Decoder level 2
        up2 = F.interpolate(dec3_out, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, enc2_out], dim=1)
        dec2_out = self.dec2(up2)

        # Decoder level 1
        up1 = F.interpolate(dec2_out, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, enc1_out], dim=1)
        dec1_out = self.dec1(up1)

        # =============================================================================
        # Output layer: Generate final RGB prediction
        # =============================================================================
        out = self.out_conv(dec1_out)  # [B, 3, H, W]
        
        return out


# =============================================================================
# Module testing code
# =============================================================================
if __name__ == "__main__":
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
