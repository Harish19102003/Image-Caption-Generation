# Image-Caption-Generation using Vision Transformer (ViT)

This project implements an end-to-end deep learning pipeline for generating captions for images using the Stanford Image Paragraph Captioning Dataset. The model uses a Vision Transformer (ViT) as an encoder to extract rich image representations and a sequence decoder (Transformer) to generate descriptive captions.

---

## Features

- Data Loading: Loads images and paragraph-style captions from the Stanford dataset.
- Preprocessing:
  - Image transformations (resize, normalization)
  - Tokenization and vocabulary building from captions
- Model Architecture:
  - Encoder: Vision Transformer (ViT) for extracting image features
  - Decoder: Sequence model (Transformer Decoder) for caption generation
- Training:
  - Cross-entropy loss
  - Teacher forcing
  - Adam optimizer
- Evaluation:
  - Caption generation and qualitative evaluation
  - Metrics support (BLEU, etc.)
- Model Utilities:
  - Save and load trained models
  - Inference pipeline
- Interactive App:
  - Gradio-based UI for real-time image caption generation

---

## Folder Structure

```
Image-caption-generation/
│
├── data/
│   ├── images/                    # Dataset images
│   └── captions/                  # Paragraph captions and metadata
├── config.py                      # Model Hyperparameters
├── model.py                       # ViT Encoder + Decoder definitions
├── get_loader.py                  # Data loading & preprocessing
├── train.py                       # Training pipeline
├── utils.py                       # Model loading & evaluation utilities
├── app.py                         # Gradio application
└── README.md                      # Project documentation
```

---

## Model Architecture

### Encoder – Vision Transformer (ViT)

Used when:
- You need global image understanding instead of local CNN features.

How it works:
- Image is split into patches
- Patches are flattened and embedded
- Positional encoding is added
- Passed through Transformer layers

Output:
```
Image → ViT → Feature Embedding Vector
```

---

### Decoder – Caption Generator

Used when:
- You want to convert image features into a text sequence.

How it works:
- Input: image features + previous words
- Output: next word prediction

Example:
```
<START> → A → dog → running → in → field
```

---

## Installation

```bash
git clone https://github.com/Harish19102003/Image-Caption-Generation.git
cd Image-Caption-Generation
pip install -r requirements.txt
```

---

##  Dataset

- **Name:** Anime Face Dataset  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/vakadanaveen/stanford-image-paragraph-captioning-dataset)  
- **Description:** Contains approximately 20k images are labelled with their corresponding paragraphs.

---

## Download Dataset
```bash
kaggle datasets download -d vakadanaveen/stanford-image-paragraph-captioning-dataset
unzip stanford-image-paragraph-captioning-dataset.zip -d data/
```

## Training

Run:

```bash
python train.py
```

This will:
- Load dataset
- Build vocabulary
- Train ViT + Decoder
- Save checkpoints

---
## To Resume training from an existing model.
```bash
python train.py --resume True
```
---

## Evaluation and Inference

Handled in `utils.py`

Run:

```python
python -m utils.py
```

---
### TensorBoard 

Run:

```bash
tensorboard --logdir tb_logs
```
---

## Gradio App

Run:

```bash
python app.py
```

Features:
- Upload an image
- Get caption instantly

Example:
```
Input: Image
Output: "A man riding a bicycle on a street"
```

---