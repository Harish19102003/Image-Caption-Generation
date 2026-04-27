# Image-Caption-Generation using Vision Transformer (ViT)

This project implements an end-to-end deep learning pipeline for generating captions for images using the Stanford Image Paragraph Captioning Dataset. The model uses a Vision Transformer (ViT) as an encoder to extract rich image representations and a Transformer Decoder to generate descriptive captions.

---

## Features

- Data Loading: Loads images and paragraph-style captions from the Stanford dataset.
- Preprocessing:
  - Image transformations (resize, crop, normalization)
  - Tokenization and vocabulary building from captions
- Model Architecture:
  - Encoder: Vision Transformer (ViT) for extracting image features
  - Decoder: Transformer Decoder for caption generation
- Training:
  - Cross-entropy loss
  - Teacher forcing
  - Adam optimizer with ReduceLROnPlateau scheduler
  - Early stopping and model checkpointing
- Evaluation:
  - Caption generation on test set
  - Metrics: BLEU-4, ROUGE-L, METEOR, CIDEr
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
├── config.py                      # Hyperparameters
├── model.py                       # ViT Encoder + Transformer Decoder
├── get_loader.py                  # Data loading & preprocessing
├── train.py                       # Training pipeline
├── utils.py                       # Evaluation utilities
├── app.py                         # Gradio application
├── Dockerfile                     # Docker build instructions
├── docker-compose.yml             # Docker run configuration
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Model Architecture

### Encoder – Vision Transformer (ViT)

Used when:
- You need global image understanding instead of local CNN features.

How it works:
- Image is split into fixed-size patches (16×16)
- Patches are flattened and projected into embeddings
- A learnable CLS token is prepended
- Positional embeddings are added
- Passed through Transformer encoder layers

Output:
```
Image → Patch Embedding → ViT Encoder → Feature Sequence
```

---

### Decoder – Caption Generator

How it works:
- Input: encoded image features + previously generated words
- Causal masking prevents attending to future tokens
- Output: next word prediction at each step

Example:
```
<START> → A → dog → running → in → the → field → <END>
```

---

## Installation

```bash
git clone https://github.com/Harish19102003/Image-Caption-Generation.git
cd Image-Caption-Generation
pip install uv 
uv pip install --system -r requirements.txt 
uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu130 
python -m spacy download en_core_web_sm
```

---

## Dataset

- **Name:** Stanford Image Paragraph Captioning Dataset
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/vakadanaveen/stanford-image-paragraph-captioning-dataset)
- **Description:** Contains approximately 19,000 images, each paired with a paragraph-length caption.

---

## Download Dataset

```bash
kaggle datasets download -d vakadanaveen/stanford-image-paragraph-captioning-dataset
unzip stanford-image-paragraph-captioning-dataset.zip -d data/
```

---

## Training

```bash
python train.py
```

This will:
- Load dataset and build vocabulary
- Train ViT Encoder + Transformer Decoder
- Log metrics to TensorBoard
- Save best checkpoint to `checkpoints/`

### Resume Training

```bash
python train.py --resume
```

---

## Evaluation and Inference

Runs the trained model on the test set and prints BLEU-4, ROUGE-L, METEOR, and CIDEr scores.

```bash
python utils.py
```

---

## TensorBoard

```bash
tensorboard --logdir tb_logs
```

---

## Gradio App

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

Features:
- Upload an image
- Get a generated paragraph caption instantly

---

## Docker

Build and run the Gradio app in a container:

```bash
docker compose up --build
```

Then open http://localhost:7860 in your browser.

On subsequent runs (no code changes):

```bash
docker compose up
```

> **Note:** GPU support requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Without it the app runs on CPU automatically.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Embedding dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Encoder/decoder layers |
| `d_ff` | 512 | Feed-forward hidden size |
| `batch_size` | 64 | Training batch size |
| `epochs` | 20 | Max training epochs |
| `lr` | 1e-4 | Learning rate |
| `dropout` | 0.1 | Dropout rate |
| `grad_clip` | 1.0 | Gradient clipping value |