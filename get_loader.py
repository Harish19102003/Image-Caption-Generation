import torch
from torch import Generator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from pathlib import Path
from collections import Counter
import spacy
import re
import numpy as np
from config import root_dir, img_dir, caption_file, batch_size, img_size

device = "cuda" if torch.cuda.is_available() else "cpu"

class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        # python -m spacy download en_core_web_sm
        self.spacy_eng = spacy.load('en_core_web_sm')

    def __len__(self):
        return len(self.itos)
    
    def clean_caption(self,text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenizer(self,text):
        text = self.clean_caption(text)
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        """Build vocabulary from the given list of sentences."""
        counter = Counter()
        for sent in sentence_list:
            for word in self.tokenizer(sent):
                counter[word] += 1

        idx = len(self.itos)
        for word, count in counter.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1


    def numericalize(self, text):
        tokens = self.tokenizer(text)

        tokens = [self.stoi["<start>"]] + [
            self.stoi.get(word, self.stoi["<unk>"]) for word in tokens
        ] + [self.stoi["<end>"]]

        return tokens
    
    def get_max_length(self, sentence_list, percentile=95):
        lengths = []

        for sent in sentence_list:
            tokens = self.numericalize(sent)  # includes <start> and <end>
            lengths.append(len(tokens))

        return int(np.percentile(lengths, percentile))
    
    def encode(self, text):
        tokens = self.numericalize(text)
        return torch.tensor(tokens)
    
    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            word = self.itos.get(idx.item(), "<unk>")

            if word in ["<pad>", "<start>", "<end>"]:
                continue

            words.append(word)

        return " ".join(words).capitalize()
    
class Build_Dataset(Dataset):
    def __init__(self,root_dir,image_dir,caption_file,Vocab,freq_threshold=5,transform=None,split=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / image_dir
        self.df = pd.read_csv(self.root_dir/caption_file)

        if split == "train":
            self.df = self.df[self.df["train"] == True].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[self.df["val"] == True].reset_index(drop=True)
        elif split == "test":
            self.df = self.df[self.df["test"] == True].reset_index(drop=True)
        elif split is None:
            self.df = self.df
        elif split is not None:
            raise ValueError("split must be one of 'train', 'val', 'test', or None")

        self.img = self.df["Image_name"].values
        self.caption = self.df["Paragraph"].values

        self.vocab = Vocab(freq_threshold)
        self.vocab.build_vocabulary(self.caption.tolist())
        self.transform = transform

    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, index):
        caption = self.caption[index]
        img_id  = f"{self.img[index]}.jpg"
        img = Image.open(self.img_dir/img_id)

        if self.transform:
            img = self.transform(img)
        
        numerical_caption = self.vocab.numericalize(caption)
        numerical_caption = torch.tensor(numerical_caption) 

        return img,numerical_caption   
    


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.cat([item[0].unsqueeze(0) for item in batch],dim=0) 

        captions = [item[1] for item in batch]

        captions = [torch.tensor(cap) for cap in captions]

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return images, captions    
    
    
g = torch.Generator()
g.manual_seed(42)

def mean_std():
    transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224,224)),
    transforms.ToTensor()])
    dataset    = Build_Dataset(root_dir, img_dir, caption_file, Vocabulary, transform=transform)
    pad_idx    = dataset.vocab.stoi["<pad>"]
    data_loader = DataLoader(dataset, batch_size=len(dataset),shuffle=False,collate_fn=MyCollate(pad_idx), generator=g)
    images = next(iter(data_loader))[0]
    images.to(device)
    mean,std = images.mean([0,2,3]), images.std([0,2,3])
    return mean,std


# already computed mean and std for the dataset to avoid recomputation every time
mean, std = [0.4695, 0.4511, 0.4148], [0.2676, 0.2635, 0.2811]

transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

dataset    = Build_Dataset(root_dir, img_dir, caption_file, Vocabulary, transform=transform)

pad_idx    = dataset.vocab.stoi["<pad>"]

train_dataset = Build_Dataset(root_dir, img_dir, caption_file, Vocabulary, transform=transform, split="train")
val_dataset   = Build_Dataset(root_dir, img_dir, caption_file, Vocabulary, transform=transform, split="val")
test_dataset  = Build_Dataset(root_dir, img_dir, caption_file, Vocabulary, transform=transform, split="test")
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate(pad_idx), generator=g)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)