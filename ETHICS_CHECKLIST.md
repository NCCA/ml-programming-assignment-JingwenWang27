# Ethics Checklist for Training Data

## Project: PBR Basecolor Prediction from Rendered Images
**Student**: Jingwen Wang  
**Student ID**: s5820023  
**Date**: January 2026

---

## 1. Dataset Information

### Dataset Name
PBR Multi-Lighting Basecolor Dataset

### Dataset Composition
- Total materials: 50 PBR ground/stone materials
- Basecolor (Ground Truth): 50 images (from material textures)
- Rendered images (Input): 300 images (50 materials × 6 lighting conditions)
- Custom test images: Personal photographs (self-captured for qualitative testing)

### Data Split
- Total dataset: 300 rendered images (50 materials × 6 lighting conditions)
- Training set: 210 images (70% of total dataset)
- Validation set: 45 images (15% of total dataset)
- Test set: 45 images (15% of total dataset)
- Split method: Random split using PyTorch's random_split function at the image level
- Limitation: Split is performed at the image level rather than material level. This means the same material may appear in training, validation, and test sets under different lighting conditions, which could lead to data leakage and potentially overestimate model performance. Future work should implement material-level splitting to better assess true generalization capability.

---

## 2. Data Sources and Licensing

### 2.1 PBR Texture Materials (Basecolor Ground Truth)
- Source: AmbientCG (https://ambientcg.com/)
- Material Type: Ground and stone surfaces (tiles, pavement, rocks, bricks)
- License: CC0 (Public Domain)
- Usage Rights: 
  - Free to use for any purpose
  - No attribution required
  - Commercial and academic use permitted
- Citation: Textures from ambientCG (https://ambientcg.com/), licensed under CC0 1.0 Universal.

### 2.2 Rendered Images (Raw Input)
- Generation Method: Self-generated using Blender 3D
- Process: 
  - Automated batch rendering using Python scripts in Blender
  - 6 different lighting setups per material
  - Consistent camera and scene configuration
- Ownership: Self-created synthetic data
- License: Own work, no licensing restrictions

### 2.3 Custom Test Images
- Source: Self-captured photographs
- Subjects: Personal objects and scenes
- Privacy: No identifiable persons or private property of others
- Consent: N/A (own photographs)

---

## 3. Ethical Considerations

### 3.1 Privacy
- No personal information: Dataset contains only synthetic 3D renders and personal photos
- No identifiable individuals: No faces or personally identifiable information
- No private property: All test images are of own belongings or public scenes

### 3.2 Consent
- CC0 materials: No consent required (public domain)
- Self-generated data: Own work, no third-party consent needed
- Personal photos: Self-captured, no consent issues

### 3.3 Bias and Fairness
- Potential biases identified:
  - Material diversity: Limited to ground/stone materials (50 variants)
  - Lighting conditions: Only 6 predefined lighting setups
  - Texture types: Primarily ground and architectural surface materials
  - Data leakage: Train/validation/test split performed at image level rather than material level, allowing the same material to appear in multiple sets under different lighting
- Mitigation strategies:
  - Selected diverse styles and colors within ground/stone category
  - Used 6 different lighting angles and intensities
  - Multi-lighting rendering to increase data diversity
- Acknowledged limitations:
  - Image-level split may lead to overestimated model performance
  - True generalization ability should be assessed on completely unseen materials

### 3.4 Environmental Impact
- Rendering energy: Blender rendering performed on local machine (NVIDIA RTX 4080)
- Carbon footprint: Minimal (batch rendering optimized for efficiency)
- Training energy: GPU training on local hardware

### 3.5 Intended Use
- Educational purpose: Academic project for ML coursework
- Research purpose: Exploring PBR basecolor prediction techniques
- Not for commercial deployment without further validation
- Not for safety-critical applications

---

## 4. Data Collection and Processing

### 4.1 Data Generation Pipeline
1. Material acquisition: Downloaded CC0 ground/stone textures from AmbientCG
2. Scene setup: Created Blender scene with standardized plane geometry
3. Lighting configuration: Defined 6 lighting setups with varying angles and intensities
4. Batch rendering: Python script to automate rendering process
5. Quality control: Manual inspection of rendered outputs

### 4.2 Data Quality Assurance
- Consistent resolution (256×256 pixels)
- Proper color space (sRGB)
- No corrupted files
- Balanced lighting conditions

### 4.3 Data Augmentation
- Training set augmentation: Random horizontal flip and random rotation (90°, 180°, 270°)
- Validation/test sets: No augmentation applied
- Primary augmentation method: Multi-lighting condition rendering
  - Each material rendered under 6 different lighting setups
  - Varying lighting angles and intensities
  - Simulates real-world lighting variations

---

## 5. Generative AI Usage Declaration

### 5.1 AI Tools Used
- Claude (Anthropic): Code assistance, debugging, documentation
- ChatGPT (OpenAI): Algorithm explanation, code optimization
- GitHub Copilot: Code completion and suggestions

### 5.2 Scope of AI Assistance
- Code structure and implementation
- PyTorch framework usage
- Documentation and comments
- Debugging and error resolution
- Unit test development
- Project organization

### 5.3 Original Contributions
- Project concept and research direction
- Dataset generation pipeline design
- Model training and evaluation
- Results analysis and interpretation

### 5.4 Citation Compliance
Following NCCA Coding Standard citation rules:
- Generative AI tools provided significant assistance throughout project development
- All AI tool usage is disclosed in this ethics checklist
- External libraries and frameworks are cited in references

Note: Per course requirements: "You may use generative AI throughout the assessment to support your own work. You do not need to acknowledge which content is generative AI-generated." Therefore, specific AI-assisted sections are not individually marked in code.

---

## 6. Reproducibility and Transparency

### 6.1 Data Availability
- PBR textures: Publicly available at https://ambientcg.com/
- Rendering scripts: Included in project repository
- Custom test images: Not publicly shared (personal photos)

### 6.2 Code Availability
- GitHub repository: Available for coursework submission
- Documentation: Comprehensive README and inline comments
- Dependencies: Listed in requirements.txt

### 6.3 Model Checkpoints
- Training checkpoints saved periodically
- Best model saved based on validation loss
- Final model saved after training completion
- Loss curves plotted and saved for analysis

---

## 7. Limitations and Future Work

### 7.1 Known Limitations
- Limited to 50 ground/stone material types
- Fixed resolution rendering (256×256)
- Synthetic data only (may not generalize well to real-world photos)
- Limited lighting diversity (6 conditions)
- Data leakage issue: Train/validation/test split performed at image level, not material level
  - Same material may appear in multiple sets under different lighting
  - Performance metrics may not accurately reflect true generalization capability
  - Model may be learning material-specific features rather than general lighting-to-basecolor mapping
- Small dataset size (300 images total)

### 7.2 Potential Risks
- Overfitting: Small dataset and data leakage may lead to overfitting
- Generalization: May not perform well on real-world photographs
- Material scope: Limited to ground/stone surfaces only
- Artistic intent: Cannot replace human artistic judgment
- Performance claims: Validation and test metrics may be overoptimistic due to data leakage

### 7.3 Future Improvements
- Implement material-level train/validation/test split to prevent data leakage
- Expand to more material categories (wood, metal, fabric)
- Include real-world photographs in training
- Test on more diverse lighting conditions
- Increase dataset size significantly
- Create a proper held-out test set with completely unseen materials
- Evaluate on standardized benchmarks

---

## 8. Compliance Checklist

- All data sources documented
- Licenses verified and compliant
- No personal/sensitive data used without consent
- Potential biases identified and acknowledged
- Data leakage issue identified and documented
- Generative AI usage fully disclosed
- NCCA coding standards followed where applicable
- Reproducibility information provided
- Limitations clearly stated

---

## 9. Declaration

I, Jingwen Wang (s5820023), declare that:

1. All training data sources used in this project are documented in this checklist
2. All publicly available datasets are used under appropriate licenses (CC0)
3. Generative AI tool usage has been fully disclosed
4. External resources are cited in the references section
5. This model is designed for educational and research purposes
6. I have made reasonable efforts to follow NCCA coding standard citation rules
7. I am aware of the limitations and potential risks of this work, including the data leakage issue in the current train/validation/test split

Special Note on AI Assistance:
- Generative AI tools (Claude, ChatGPT, GitHub Copilot) provided extensive assistance in code development, debugging, and documentation
- Per course requirements: "You may use generative AI throughout the assessment to support your own work. You do not need to acknowledge which content is generative AI-generated."
- The core project concept, data generation pipeline, model training strategy, and evaluation were designed and executed by myself
- All code has been reviewed, understood, and tested by myself

Signature: Jingwen Wang  
Date: 14 January 2026  
Course: Software Engineering for Media (Level 7)  
Institution: National Centre for Computer Animation, Bournemouth University

---

## 10. References

### Data Sources
- AmbientCG. (2024). Free PBR Materials. Retrieved from https://ambientcg.com/
- License: CC0 1.0 Universal (Public Domain)

### Software and Frameworks
- Blender Foundation. (2024). Blender 3D. https://www.blender.org/
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32.
- PyTorch. (2024). PyTorch Deep Learning Framework. https://pytorch.org/

### AI Tools
- Anthropic. (2024). Claude AI Assistant. https://www.anthropic.com/
- OpenAI. (2024). ChatGPT. https://openai.com/chatgpt
- GitHub. (2024). GitHub Copilot. https://github.com/features/copilot

### Related Work
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. ECCV.
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556. (Used for perceptual loss)

---

End of Ethics Checklist