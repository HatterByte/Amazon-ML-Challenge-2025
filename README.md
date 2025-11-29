# Amazon ML Challenge 2025 - Solution 

**Public Leaderboard: #1 with SMAPE Score of 39.1969**

![Leaderboard](assets/leaderboard.png)

## Overview

This repository contains our winning solution for the Amazon ML Challenge 2025, where we developed a state-of-the-art multimodal regression pipeline for price prediction in visual catalogs. Our approach leverages deep learning to fuse image and text embeddings for accurate price estimation.

## Problem Statement

The challenge involved predicting product prices from multimodal data (images + text descriptions) in an e-commerce catalog setting. This required understanding both visual product features and textual descriptions to estimate fair market prices.

## Solution Architecture

Our winning solution is based on a **triple-model ensemble** that combines three distinct approaches:

### Core Methodology

- **Multimodal Deep Learning System** integrating image and text embeddings
- **Fine-tuned CLIP Models** for superior image-text feature fusion
- **Custom Multi-layer Regression Head** optimized for price prediction
- **Grid-searched Ensemble Weighting** for optimal model combination

## Approaches Implemented

### Approach 1: XGBoost Baseline (`Approach_1_XGBoost.py`)

- Traditional gradient boosting approach
- Feature engineering with text embeddings using SentenceTransformers
- Brand extraction using spaCy NLP pipeline
- Serves as a robust baseline for comparison

### Approach 2: EfficientNet-BERT (`Approach_2_Efficientnet_bert.ipynb`)

- **Image Features**: EfficientNet-B4 for visual feature extraction
- **Text Features**: BERT embeddings for textual understanding
- **Fusion**: Concatenated multimodal features with regression head
- Strong performance on both modalities independently

### Approach 3: OpenAI CLIP ViT-Large (`Approach_3_Openai_Clip-Vit-Large.py`)

- **Model**: OpenAI CLIP ViT-L/14
- **Strategy**: Fine-tuning pre-trained CLIP for price regression
- **Architecture**: Custom regression head on top of CLIP embeddings
- **Optimization**: Differential learning rates for CLIP backbone vs. head

### Approach 4: LAION CLIP ViT-H (`Approach_4_Laion_Clip_Vith.py`)

- **Model**: LAION CLIP ViT-H/14 (larger variant)
- **Enhanced Capacity**: Higher parameter count for better feature learning
- **Training**: Specialized for large-scale visual-linguistic understanding

### Final Solution: Triple Ensemble (`Ensemble_Model.py`)

- **Combination**: CLIP ViT-H + CLIP ViT-L + EfficientNet-BERT
- **Weighting**: Grid-searched optimal weights for each model
- **Performance Gain**: 1-2% SMAPE reduction over single-model baselines
- **Robustness**: Leverages complementary strengths of each approach

## Key Results
- **Final SMAPE Score**: 39.1969

## Technical Implementation

### Model Architecture Details

#### CLIP-based Models

```
Input: [Image, Text] → CLIP Encoder → Multimodal Embeddings → Regression Head → Price
```

#### EfficientNet-BERT

```
Image → EfficientNet-B4 → Image Features ↘
                                          → Concatenation → MLP → Price
Text → BERT → Text Features              ↗
```

### Key Technical Features

- **Mixed Precision Training** for faster convergence
- **Differential Learning Rates** for pre-trained vs. new layers
- **Advanced Data Augmentation** for robustness
- **Early Stopping** with validation monitoring
- **Grid Search Optimization** for ensemble weights

## Project Structure

```
Amazon ML Challenge/
├── Approach_1_XGBoost.py              # XGBoost baseline approach
├── Approach_2_Efficientnet_bert.ipynb # EfficientNet + BERT model
├── Approach_3_Openai_Clip-Vit-Large.py # OpenAI CLIP ViT-L/14
├── Approach_4_Laion_Clip_Vith.py      # LAION CLIP ViT-H/14
├── Ensemble_Model.py                   # Final ensemble solution
├── ML_Challenge_Data_Cleaning.ipynb    # Data preprocessing
├── assets/
│   └── leaderboard.png                # Final ranking proof
├── src/
│   ├── image_downloader.ipynb         # Data acquisition utilities
│   └── utils.py                       # Utility functions
└── README.md                          # This file
```

## Installation & Setup

### Prerequisites

```bash
pip install torch torchvision
pip install transformers
pip install sentence-transformers
pip install xgboost
pip install pandas numpy scikit-learn
pip install pillow
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage

### Training Individual Models

```bash
# EfficientNet-BERT
python Approach_2_Efficientnet_bert.py
# OpenAI CLIP ViT-Large
python Approach_3_Openai_Clip-Vit-Large.py
# LAION CLIP ViT-H
python Approach_4_Laion_Clip_Vith.py
```

### Running the Ensemble

```bash
python Ensemble_Model.py
```

### Data Preprocessing

```bash
jupyter notebook ML_Challenge_Data_Cleaning.ipynb
```

## Key Learnings & Insights

1. **Multimodal Fusion**: CLIP models excel at understanding image-text relationships
2. **Ensemble Benefits**: Combining diverse architectures provides robust predictions
3. **Fine-tuning Strategy**: Differential learning rates crucial for pre-trained models
4. **Data Quality**: Proper preprocessing significantly impacts model performance
5. **Model Diversity**: Different architectures capture complementary patterns

## Competition Highlights

- **Innovative Approach**: First to successfully ensemble CLIP variants with traditional models
- **Technical Excellence**: Advanced multimodal deep learning implementation
- **Robust Solution**: Consistent performance across diverse product categories
- **Scalable Architecture**: Modular design allowing easy experimentation

## Future Improvements

- **Vision Transformers**: Experiment with newer ViT architectures
- **Advanced Ensembling**: Implement stacking or meta-learning approaches
- **Data Augmentation**: Explore domain-specific augmentation techniques
- **Model Compression**: Optimize for production deployment
