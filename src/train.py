"""
Training Script - PBR Basecolor Prediction Model


Supports multi-lighting condition training

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SimpleUNet
from dataset import PBRBasecolorDataset
from perceptual_loss import PerceptualLoss


def train():
    """Main training function """
    
    # =============================================================================
    # Training configuration
    # =============================================================================
    raw_dirs = [
        "data/raw_light1",
        "data/raw_light2", 
        "data/raw_light3",
        "data/raw_light4",
        "data/raw_light5",
        "data/raw_light6",
    ]
    basecolor_dir = "data/basecolor"
    image_size = 256
    batch_size = 8
    num_epochs = 300
    learning_rate = 1e-4
    checkpoint_dir = "checkpoints"
    
    # =============================================================================
    # Device setup: Use GPU if available
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # =============================================================================
    # Create dataset and split into train/validation sets
    # =============================================================================
    print(f"\nLoading dataset...")
    dataset = PBRBasecolorDataset(
        raw_dirs=raw_dirs,
        basecolor_dir=basecolor_dir,
        image_size=image_size,
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset loaded successfully")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    # =============================================================================
    # Initialize model with increased capacity
    # =============================================================================
    print(f"\nCreating model...")
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
    ).to(device)
    
    print(f"Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # =============================================================================
    # Define loss function and optimizer
    # =============================================================================
    criterion = PerceptualLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=20,
        verbose=True
    )
    
    # =============================================================================
    # Create checkpoint directory
    # =============================================================================
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # Training loop
    # =============================================================================
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Loss function: Perceptual Loss (30% Pixel + 70% VGG)")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # =============================================================================
        # Training phase
        # =============================================================================
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for raw, basecolor in train_bar:
            raw = raw.to(device)
            basecolor = basecolor.to(device)
            
            # Forward pass
            outputs = model(raw)
            loss = criterion(outputs, basecolor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # =============================================================================
        # Validation phase
        # =============================================================================
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
            for raw, basecolor in val_bar:
                raw = raw.to(device)
                basecolor = basecolor.to(device)
                
                outputs = model(raw)
                loss = criterion(outputs, basecolor)
                
                val_loss += loss.item()
                val_bar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # =============================================================================
        # Print epoch statistics
        # =============================================================================
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # =============================================================================
        # Save best model based on validation loss
        # =============================================================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(checkpoint_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            print(f"  Saved best model: {checkpoint_path}")
        
        # =============================================================================
        # Save periodic checkpoints every 50 epochs
        # =============================================================================
        if (epoch + 1) % 50 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        print()
   
    # =============================================================================
    # Save final model after training completes
    # =============================================================================
    final_path = Path(checkpoint_dir) / "final_model.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model saved at: {final_path}")
    print(f"Best model saved at: {Path(checkpoint_dir) / 'best_model.pth'}")
    
    # =============================================================================
    # Plot and save training curves
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Perceptual Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(checkpoint_dir) / 'loss_curve.png', dpi=150, bbox_inches='tight')
    print(f"Loss curve saved at: {Path(checkpoint_dir) / 'loss_curve.png'}")


if __name__ == "__main__":
    train()
