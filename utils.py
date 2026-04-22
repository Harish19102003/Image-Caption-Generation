import torch
import os
from torchtext.data.metrics import bleu_score
from model import device
from train import trainer, load_model
from get_loader import dataset, test_loader
from config import model_path
import warnings

if device.type == "cuda":
    torch.set_float32_matmul_precision( 'high')

def clean(tokens):
    """Remove special tokens from token list."""
    return [
        t for t in tokens
        if t not in ["<pad>", "<start>", "<end>", "<unk>"]
    ]

def main():
    warnings.filterwarnings("ignore")
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
        

    model = load_model(model_path).eval()

    # ── bulk predictions ──
    preds = trainer.predict(model, test_loader)
    preds = [seq for batch in preds for seq in batch]  # type: ignore
    # preds: list of strings e.g. "select count ( * ) from students"

    # ── tokenize predictions ──
    pred_tokens = [clean(dataset.vocab.tokenizer(seq))for seq in preds]

    # ── tokenize references using itos directly ──
    ref_tokens = [clean([dataset.vocab.itos[idx.item()]for idx in trg])for _, trg in test_loader.dataset]

    # ── compute corpus BLEU ──
    score = bleu_score(pred_tokens, [[ref] for ref in ref_tokens])
    print(f"BLEU: {score:.4f} ({score*100:.2f}%)")

if __name__ == "__main__":
    main()