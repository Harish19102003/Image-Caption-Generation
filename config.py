from pathlib import Path

root_dir = Path("data/stanford Image Paragraph Captioning dataset")
caption_file = "stanford_df_rectified.csv"
img_dir = "stanford_images"
output_dir = Path("checkpoints")
output_file = "img_caption_model"
model_path = Path(f"{output_dir}/{output_file}.ckpt")

# Hyperparameters
img_size = 224
d_model = 256
n_heads = 8
n_layers = 3
d_ff = 2 * d_model
dropout = 0.1
max_len = 500
lr = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
epochs = 50
batch_size = 32