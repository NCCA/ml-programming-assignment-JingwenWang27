# Deep Learning-Based Texture De-Lighting System

This project removes lighting effects (shadows, highlights, etc.) from material texture photos and predicts a cleaner BaseColor (albedo) map, reducing lighting contamination in PBR material workflows.

---

## Project Overview

**Problem:** Material textures captured under different lighting conditions often contain shadows and specular highlights. These lighting artifacts can harm PBR authoring, causing materials to respond incorrectly to lighting in 3D scenes.

**Solution:** A deep learning model (U-Net with perceptual loss using VGG16 features) learns a mapping from lit textures to clean base color.

**Use cases:** PBR material creation for games/film, texture cleanup, scan post-processing, and asset standardization.

---
## Model Weights

Download the pre-trained model checkpoint:

**Checkpoint (Epoch 200):**
- Google Drive: [checkpoint_epoch_200.pth](https://drive.google.com/uc?export=download&id=16KOjsu25S74SmWW5qciOpZKDW3i-hWmh)
- Size: ~120 MB
- Training: 200 epochs on PavingStones dataset

ml-programming-assignment-JingwenWang27/
└── checkpoints/
    └── checkpoint_epoch_200.pth


**Quick download:**
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download using gdown (recommended)
pip install gdown
gdown 16KOjsu25S74SmWW5qciOpZKDW3i-hWmh -O checkpoints/checkpoint_epoch_200.pth

# Or download manually from the link above



## Core Methods

### Model Architecture

- U-Net encoder-decoder (4 levels) with skip connections to preserve high-frequency texture details
- Base channels: 128; ~31M parameters; 3×3 conv, BatchNorm + ReLU

### Loss Function

```text
Perceptual Loss = 0.3 × Pixel Loss + 0.7 × VGG16 Feature Loss
├─ Pixel Loss = 0.5 × L1 + 0.5 × MSE
└─ Feature Loss = MSE on VGG16 relu3_3 features
```

**Notes:** VGG16 is frozen; relu3_3 features balance low-level texture detail and mid-level structure.

### Training Setup

- **Optimizer:** Adam (lr=0.0001)
- **LR scheduler:** ReduceLROnPlateau (patience=10, factor=0.5)
- **Batch size:** 8; epochs: 300
- **Augmentation:** random flips, rotations

### Dataset

- **Source:** ambientCG open material library, PavingStones series
- **Scale:** 50 textures × 6 lighting conditions = 300 samples (270 train / 30 validation)
- **Preprocessing:** 1024×1024 → 256×256 (bilinear resize), normalized to [0, 1]

**Directory structure:**

```text
data/
├── basecolor/      # BaseColor (Ground Truth)
├── raw_light1/     # Lighting condition 1
├── raw_light2/     # Lighting condition 2
├── raw_light3/     # Lighting condition 3
├── raw_light4/     # Lighting condition 4
├── raw_light5/     # Lighting condition 5
└── raw_light6/     # Lighting condition 6
```

---

## Usage

### Installation

```bash
git clone https://github.com/NCCA/ml-programming-assignment-JingwenWang27.git
cd ml-programming-assignment-JingwenWang27
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training

```bash
python src/train.py
```

**Outputs:**
- Best model: `checkpoints/best_model.pth`
- Periodic checkpoints: saved every 10 epochs
- Logs: losses printed during training

### Inference

```bash
# Single image
python src/infer.py --input data/raw_light1/PavingStones101_Raw_L1.png --output results/output.png

# Batch inference
python src/infer.py --input data/raw_light1/ --output results/
```

### Visualization

```bash
jupyter notebook demo.ipynb
```

The notebook includes input/output comparisons, cross-lighting comparisons, and loss curve analysis.

### Testing

```bash
pytest tests/ -v
```

---

## Limitations

- Trained only on PavingStones (stone materials)
- Resolution is limited to 256×256
- Domain gap exists on real photographs

---

## Future Work

- **Higher resolution:** train at 512×512 or 1024×1024
- **More material categories:** wood, metal, fabric, etc.
- **Real photo adaptation:** fine-tune on real-world data
- **Super-resolution:** generate 2K/4K textures
- **Other PBR maps:** extend to Roughness, Metallic, Normal

---

## Technical Details

### U-Net Overview

```text
Input (3, 256, 256)
  ↓ Encoder: 3→128→256→512→1024
  ↓ Bottleneck: 1024→2048
  ↓ Decoder: 2048→1024→512→256→128 (with skip connections)
Output (3, 256, 256)
```

---

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* arXiv:1603.08155.
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.

---

## Project Information

- **Coursework:** Software Engineering for Media (Level 7)
- **School:** NCCA, Bournemouth University
- **Student:** Jingwen Wang (5820023)
- **Repository:** https://github.com/NCCA/ml-programming-assignment-JingwenWang27
- **Ethics statement:** See `ETHICS_CHECKLIST.md` for the full ethics review and AI usage disclosure.
- **Last updated:** January 2026 | **Status:** Completed and submitted for assessment