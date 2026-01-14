import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):

    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder: Extract features layer by layer
        self.enc1 = self._block(in_channels, base_channels)           # 3 → 64 
        self.enc2 = self._block(base_channels, base_channels * 2)     # 64 → 128 
        self.enc3 = self._block(base_channels * 2, base_channels * 4) # 128 → 256 
        self.enc4 = self._block(base_channels * 4, base_channels * 8) # 256 → 512 
        
        # Bottleneck: Most abstract features
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)  # 512 → 1024 
        
        # Decoder: Recover details (channels include skip connections)
        self.dec4 = self._block(base_channels * 16 + base_channels * 8, base_channels * 8)  # (1024+512) → 512
        self.dec3 = self._block(base_channels * 8 + base_channels * 4, base_channels * 4)   # (512+256) → 256
        self.dec2 = self._block(base_channels * 4 + base_channels * 2, base_channels * 2)   # (256+128) → 128
        self.dec1 = self._block(base_channels * 2 + base_channels, base_channels)           # (128+64) → 64
        
        # Output layer: Convert features to RGB
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def _block(self, in_c, out_c):
        
        #Basic convolution block: Double convolution
        
        #Structure: Conv(3x3) → BatchNorm → ReLU → Conv(3x3) → BatchNorm → ReLU
        #Function: 
        
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),  # Convolution
            nn.BatchNorm2d(out_c),                                          #  Batch Normalization
            nn.ReLU(inplace=True),                                          # Activation
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False), #  Second convolution
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        
        #Forward propagation
        
        #Args: x: Input image [Batch, 3, 256, 256]
        
        #Returns:Predicted basecolor [Batch, 3, 256, 256]
     
        
        #  Encoder path: Extract multi-scale features
        enc1_out = self.enc1(x)                          # [B, 64, 256, 256] 
        enc2_out = self.enc2(F.max_pool2d(enc1_out, 2))  # [B, 128, 128, 128] Downsample 2x
        enc3_out = self.enc3(F.max_pool2d(enc2_out, 2))  # [B, 256, 64, 64] Downsample 4x
        enc4_out = self.enc4(F.max_pool2d(enc3_out, 2))  # [B, 512, 32, 32] Downsample 8x
        
        # Bottleneck: Process most abstract features
        bottleneck_out = self.bottleneck(F.max_pool2d(enc4_out, 2))  # [B, 1024, 16, 16] 
        
        # Decoder path: Upsample and fuse encoder features
        
        # Decoder level 4: Upsample + skip connection
        up4 = F.interpolate(bottleneck_out, scale_factor=2, mode="bilinear", align_corners=False)
        up4 = torch.cat([up4, enc4_out], dim=1)  #  Concatenate encoder features
        dec4_out = self.dec4(up4)                # [B, 512, 32, 32]
        
        # Decoder level 3
        up3 = F.interpolate(dec4_out, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = torch.cat([up3, enc3_out], dim=1)
        dec3_out = self.dec3(up3)                # [B, 256, 64, 64]
        
        # Decoder level 2
        up2 = F.interpolate(dec3_out, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, enc2_out], dim=1)
        dec2_out = self.dec2(up2)                # [B, 128, 128, 128]
        
        # Decoder level 1
        up1 = F.interpolate(dec2_out, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, enc1_out], dim=1)
        dec1_out = self.dec1(up1)                # [B, 64, 256, 256]
        
        # Output layer: Generate final RGB prediction
        out = self.out_conv(dec1_out)            # [B, 3, 256, 256]
        
        return out
