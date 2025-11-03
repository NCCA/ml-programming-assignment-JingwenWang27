# Jingwen Wang s5820023 ML Project

## Introduction

This project aims to develop a machine learning tool that generates PBR material maps from single photographs, addressing the lighting contamination issue in existing scanned material libraries. The focus will be on a single material category, with the first stage targeting either extracting base color by removing lighting effects, or implementing photo super-resolution.

## Main Approach

**Option 1:** Pixel classification - classify pixels into diffuse, specular highlight, and shadow regions to extract base color

**Option 2:** Super-resolution - upscale low-resolution photos to 2K/4K quality

Feedback needed on which approach is more suitable for the first stage.

## Key Datasets

- Segmentation: UCI Image Segmentation Dataset
- Super-resolution: DIV2K, Urban100

## Reading Material

- Minaee, Shervin, et al. Image Segmentation Using Deep Learning: A Survey. arXiv, 2020.
- Wang, Zhihao, et al. Deep Learning for Image Super-Resolution: A Survey. arXiv, 2020.