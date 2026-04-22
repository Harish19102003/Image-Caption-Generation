import torch
import os 
import time
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from config import epochs, grad_clip, output_dir, output_file, model_path
from get_loader import train_loader, val_loader
from model import model
import warnings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.set_float32_matmul_precision( 'high')

early_stop = EarlyStopping(
        monitor  = 'val_loss',
        patience = 5,        # stop if val_loss doesn't improve for 5 epochs
        mode     = 'min'
    )

def load_model(output_file, model=model):
    checkpoint = torch.load(output_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


checkpoint = ModelCheckpoint(
    dirpath= output_dir,
    monitor   = 'val_loss',
    save_top_k = 1,          # only keep best model
    mode      = 'min',
    filename  = output_file,
    auto_insert_metric_name=False
    )

lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger("tb_logs", name=output_file)
trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=grad_clip,accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1, callbacks= [early_stop, checkpoint,lr_monitor], logger=logger)

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train the image captioning model.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    args = parser.parse_args()

    ckpt = None
    if args.resume:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        print(f"Resuming from {model_path}...")
        ckpt = model_path

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
    print(f"Training time: {(time.time() - start) / 60:.1f} min")

if __name__ == "__main__":
    main()