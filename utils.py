import torch
import os
from torchtext.data.metrics import bleu_score
from  rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from model import VisionTransformer, device
import pytorch_lightning as pl
from get_loader import dataset, get_loaders
from config import model_path, img_size, d_model, n_heads, n_layers, d_ff, dropout, max_len
import warnings

_, _, test_loader = get_loaders()

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

    model = VisionTransformer.load_from_checkpoint(
    model_path,
    input_dim=len(dataset.vocab),
    img_size=img_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len,
    dataset=dataset,
    ).eval().to(device)
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1)

    preds = trainer.predict(model, test_loader)
    preds = [seq for batch in preds for seq in batch]  # type: ignore

    pred_tokens = [clean(dataset.vocab.tokenizer(seq)) for seq in preds]
    ref_tokens  = [
        clean([dataset.vocab.itos[idx] for idx in trg.tolist()])
        for _, trg in test_loader.dataset
    ]

    bleu = bleu_score(pred_tokens, [[ref] for ref in ref_tokens])
    print(f"BLEU-4:  {bleu:.4f} ({bleu*100:.2f}%)")

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [
        scorer.score(" ".join(r), " ".join(p))['rougeL'].fmeasure
        for p, r in zip(pred_tokens, ref_tokens)
    ]
    RouGE_L = sum(rouge_scores) / len(rouge_scores)
    print(f"ROUGE-L: {RouGE_L:.4f} ({RouGE_L*100:.2f}%)")

    # METEOR
    meteor_scores = [
        meteor_score([r], p)
        for p, r in zip(pred_tokens, ref_tokens)
    ]
    METEOR = sum(meteor_scores) / len(meteor_scores)
    print(f"METEOR: {METEOR:.4f} ({METEOR*100:.2f}%)")

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(
        {i: [" ".join(r)] for i, r in enumerate(ref_tokens)},
        {i: [" ".join(p)] for i, p in enumerate(pred_tokens)}
    )
    print(f"CIDEr: {cider_score:.4f}")

if __name__ == "__main__":
    main()