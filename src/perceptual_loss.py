import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):

    
    def __init__(self):
        super().__init__()
        
        # =============================================================================
        # Load pretrained VGG16 and extract feature layers
        # =============================================================================
        vgg = models.vgg16(pretrained=True)
        
        # Use first 16 layers (up to relu3_3) for feature extraction
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        # Freeze VGG parameters - no training needed

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # =============================================================================
        # Define pixel-level loss functions
        # =============================================================================
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Calculate combined perceptual and pixel loss
        
        Args:
            pred: Predicted image [B, 3, H, W]

            target: Ground truth image [B, 3, H, W]
        
        Returns:
            total_loss: Combined loss value

        """
        
        # =============================================================================
        # Pixel-level loss: Ensure color accuracy
        # =============================================================================
        pixel_loss = 0.5 * self.l1_loss(pred, target) + 0.5 * self.mse_loss(pred, target)
        
        # =============================================================================
        # Perceptual loss: Ensure texture and structure similarity
        # =============================================================================
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        perceptual_loss = self.mse_loss(pred_features, target_features)
        
        # =============================================================================
        # Combine losses: 30% pixel + 70% perceptual
        # =============================================================================
        total_loss = 0.3 * pixel_loss + 0.7 * perceptual_loss
        
        return total_loss


# =============================================================================
# Module testing code
# =============================================================================
if __name__ == "__main__":
    criterion = PerceptualLoss()
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    loss = criterion(pred, target)
    print(f"Loss: {loss.item()}")
