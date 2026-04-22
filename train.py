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

output_dir = output_dir
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
    parser.add_argument("--resume", action="store_true",default=False, help="Resume training from an existing model.")
    args = parser.parse_args()
    if args.resume :
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path} for continued training...")
            model = load_model(model_path)
            trainer.fit(model, train_loader, val_loader)
        else:
            print("No existing model found or training flag not set")
    
    else:
        if not os.path.exists(model_path):   
            output_dir.mkdir(parents=True, exist_ok=True)
        from model import model
        start = time.time() // 60
        trainer.fit(model, train_loader, val_loader)
        end = time.time() // 60
        print(f"Training time: {end - start}")

if __name__ == "__main__":
    main()