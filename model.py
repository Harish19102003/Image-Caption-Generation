import math
from numpy import size
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from get_loader import dataset, transform
from config import img_size, d_model, n_heads, n_layers, d_ff, dropout, max_len, lr, weight_decay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WordEmbedding(nn.Module):
    def __init__(self,input_dim:int,d_model: int,):
        super().__init__()
        self.embedding = nn.Embedding(input_dim,d_model)
        self.scale     = math.sqrt(d_model)
    def forward(self,x):
        return self.embedding(x) * self.scale
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=224, in_channels=3, embed_dim=512,dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size) #(B, c , H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2) #(B, embed_dim, H/patch_size * W/patch_size) -> (B, num_patches, embed_dim)
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """
    Adds position information to embeddings.
    Transformer has no recurrence so it needs to know token order explicitly.

    sin/cos waves of different frequencies:
      pos 0: [sin(0), cos(0), sin(0), cos(0), ...]
      pos 1: [sin(1), cos(1), sin(0.1), cos(0.1), ...]
      pos 2: [sin(2), cos(2), sin(0.2), cos(0.2), ...]
    Each position gets a unique fingerprint.
    """
    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe: [max_len, d_model]
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term: [d_model/2] — different frequency per dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)   # [1, max_len, d_model] — broadcast over batch
        self.register_buffer('pe', pe)   # not a parameter, but saved with model


    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.shape[1], :] #type: ignore
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x))) # Residual connection around attention and dropout
        x = x + self.dropout(self.mlp(self.norm2(x)))  # Residual connection around MLP and dropout
        return x
    
class ViTEncoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, 3, d_model, dropout=dropout)

        n_patches = (img_size // patch_size) ** 2

        self.norm = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))

        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_model * 4) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.patch_embed(x)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed

        for layer in self.encoder:
            x = layer(x)

        return self.norm(x)

class VisionTransformer(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 img_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 max_len: int = 200,
                 dataset= dataset,
                 pad_idx: int = 0):
        super().__init__()
        self.dataset   = dataset
        self.pad_idx   = pad_idx
        self.d_model   = d_model
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.trg_embedding = WordEmbedding(input_dim, d_model)
        self.pos_enc       = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = ViTEncoder(img_size, patch_size=16, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True),
            num_layers=n_layers)
        self.fc_out    = nn.Linear(d_model, input_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def make_masks(self, src, trg):
        """Create padding masks and causal mask."""
        trg_pad = (trg == self.pad_idx)
        trg_mask = torch.triu(
            torch.ones(trg.shape[1], trg.shape[1], device=trg.device), diagonal=1
        ).bool()
        return trg_mask, trg_pad

    def forward(self, image, trg_tokens):
        memory = self.encoder(image)
        trg = self.pos_enc(self.trg_embedding(trg_tokens))
        
        # Create masks
        trg_mask = self._make_tgt_mask(trg.shape[1], trg.device)
        trg_pad = (trg_tokens == self.pad_idx)
        
        output = self.decoder(
            trg,
            memory,
            tgt_mask=trg_mask,
            tgt_key_padding_mask=trg_pad,
        )
        return self.fc_out(output)
    
    def configure_optimizers(self):  # type: ignore
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_loss",
                "interval":  "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        src, trg = batch
        output   = self(src, trg[:, :-1])
        output   = output.reshape(-1, output.shape[-1])
        target   = trg[:, 1:].reshape(-1)

        loss = self.criterion(output, target)
        mask = target != self.pad_idx
        acc  = (output.argmax(dim=1)[mask] == target[mask]).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # loss (keep this)
        output = self(images, targets[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = targets[:, 1:].reshape(-1)

        loss = self.criterion(output, target)
        mask = target != self.pad_idx
        acc  = (output.argmax(dim=1)[mask] == target[mask]).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def _make_tgt_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def predict_step(self, batch, batch_idx, max_len=100):
        src, _     = batch
        batch_size = src.shape[0]
        end_idx    = self.dataset.vocab.stoi["<end>"]
        start_idx  = self.dataset.vocab.stoi["<start>"]

        with torch.no_grad():
            memory  = self.encoder(src)

        trg      = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            trg_mask = self._make_tgt_mask(trg.shape[1], src.device)
            trg_pad  = (trg == self.pad_idx)

            with torch.no_grad():
                trg_emb = self.pos_enc(self.trg_embedding(trg))
                output  = self.decoder(trg_emb,
                    memory,
                    tgt_mask=trg_mask
                )
                output = self.fc_out(output)

            next_tok = output[:, -1, :].argmax(dim=1, keepdim=True)
            trg      = torch.cat([trg, next_tok], dim=1)
            finished |= (next_tok.squeeze(1) == end_idx)
            if finished.all():
                break

        results = []
        for i in range(batch_size):
            tokens = trg[i, 1:].tolist()
            if end_idx in tokens:
                tokens = tokens[:tokens.index(end_idx)]
            sql = " ".join(
                self.dataset.vocab.itos[t]
                for t in tokens
                if t not in [self.pad_idx,
                             self.dataset.vocab.stoi["<start>"],
                             self.dataset.vocab.stoi["<unk>"]]
            )
            results.append(sql)
        return results
    
    def tokens_to_text(self, tokens):
        words = []
        for t in tokens:
            if t == self.dataset.vocab.stoi["<end>"]:
                break
            if t not in [self.pad_idx,
                        self.dataset.vocab.stoi["<start>"],
                        self.dataset.vocab.stoi["<unk>"]]:
                words.append(self.dataset.vocab.itos[t])
        return words
    
    def beam_search(self, image, beam_size=5, max_len=50, length_penalty=0.7):
        """
        image: (1, C, H, W) — single image tensor, already transformed
        beam_size: number of beams to keep
        length_penalty: >1 favors longer sequences, <1 favors shorter
        """
        self.eval()
        device = image.device

        start_idx = self.dataset.vocab.stoi["<start>"]
        end_idx   = self.dataset.vocab.stoi["<end>"]

        with torch.no_grad():
            # Encode image once, then expand for all beams
            memory = self.encoder(image)                          # (1, N, D)
            memory = memory.expand(beam_size, -1, -1)            # (beam_size, N, D)

            # Each beam: (score, token_sequence)
            beams = [(0.0, [start_idx])]
            completed = []

            for _ in range(max_len):
                candidates = []

                for score, tokens in beams:
                    # Don't expand already-finished beams
                    if tokens[-1] == end_idx:
                        completed.append((score, tokens))
                        continue

                    trg = torch.tensor([tokens], device=device)               # (1, seq_len)
                    trg = trg.expand(beam_size, -1)                           # (beam_size, seq_len)

                    trg_emb  = self.pos_enc(self.trg_embedding(trg))
                    trg_mask = self._make_tgt_mask(trg.shape[1], device)

                    output = self.decoder(trg_emb, memory, tgt_mask=trg_mask)
                    logits = self.fc_out(output)                              # (beam_size, seq_len, vocab)

                    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)  # (vocab_size,)

                    # Take top beam_size next tokens
                    topk_log_probs, topk_tokens = log_probs.topk(beam_size)

                    for log_prob, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                        new_score = score + log_prob
                        candidates.append((new_score, tokens + [token]))

                if not candidates:
                    break

                # Length-penalized score for ranking
                def penalized(item):
                    s, toks = item
                    return s / (len(toks) ** length_penalty)

                # Keep top beam_size candidates
                candidates.sort(key=penalized, reverse=True)
                beams = candidates[:beam_size]

                # Stop if all beams have ended
                if all(t[-1] == end_idx for _, t in beams):
                    completed.extend(beams)
                    break

            # Pick best completed sequence, fall back to best beam
            final = completed if completed else beams
            final.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            best_tokens = final[0][1]

        text = self.tokens_to_text(best_tokens)
        return " ".join(text).capitalize()
    

    def generate_caption(self, image, max_len=max_len, beam_size=5):
        image = transform(image).to(device).unsqueeze(0)
        self.eval()
        if beam_size == 1:
            with torch.no_grad():
                memory = self.encoder(image)

                start_idx = self.dataset.vocab.stoi["<start>"]
                end_idx   = self.dataset.vocab.stoi["<end>"]

                trg = torch.tensor([[start_idx]], device=image.device)

                for _ in range(max_len):
                    trg_emb = self.pos_enc(self.trg_embedding(trg))
                    trg_mask = self._make_tgt_mask(trg.shape[1], image.device)

                    output = self.decoder(trg_emb, memory, tgt_mask=trg_mask)
                    output = self.fc_out(output)

                    next_token = output[:, -1, :].argmax(dim=1, keepdim=True)
                    trg = torch.cat([trg, next_token], dim=1)

                    if next_token.item() == end_idx:
                        break
            
            tokens = trg.squeeze(0).tolist()
            text = self.tokens_to_text(tokens)
            return " ".join(text).capitalize()
        else:
            return self.beam_search(image, beam_size=beam_size, max_len=max_len)

model = VisionTransformer(
    input_dim=len(dataset.vocab),
    img_size=img_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len,
)
model = model.to(device)