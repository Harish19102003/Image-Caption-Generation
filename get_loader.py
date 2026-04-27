import pickle
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
from config import root_dir, img_dir, caption_file, batch_size, img_size, augment

device = "cuda" if torch.cuda.is_available() else "cpu"

g = Generator()
g.manual_seed(42)

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
    
    def save_vocabulary(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


    def load_vocabulary(self, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
class Build_Dataset(Dataset):
    def __init__(self,root_dir,image_dir,caption_file,build_vocab=False,img_size=224,freq_threshold=5,augment=None,split=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / image_dir
        self.df = pd.read_csv(self.root_dir/caption_file)
        # build vocabulary from the entire dataset to ensure all words are included, even if some splits have fewer examples
        self.vocab_caption = self.df["Paragraph"].values
        self.vocab = Vocabulary(freq_threshold)
        if build_vocab:
            self.vocab.build_vocabulary(self.vocab_caption.tolist())
            self.vocab.save_vocabulary(self.root_dir / "vocab.pkl")
        else:
            self.vocab = self.vocab.load_vocabulary(self.root_dir / "vocab.pkl")
        self.img_size = img_size
        # already computed mean and std for the dataset to avoid recomputation every time
        self.mean = [0.4695, 0.4511, 0.4148]
        self.std = [0.2430, 0.2382, 0.2423]
        if augment:
            # With augmentations (flip + color jitter)
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            # Without augmentations
            self.transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])

        if split == "train":
            self.df = self.df[self.df["train"] == True].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[self.df["val"] == True].reset_index(drop=True)
        elif split == "test":
            self.df = self.df[self.df["test"] == True].reset_index(drop=True)
        elif split is None:
            self.df = self.df
        else:
            raise ValueError("split must be one of 'train', 'val', 'test', or None")

        self.img = self.df["Image_name"].values
        self.caption = self.df["Paragraph"].values
        
    def mean_and_std(self):
        """Compute the mean and std of the dataset."""
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for img_name in self.img:
            img_path = self.img_dir / f"{img_name}.jpg"
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img)
            mean += img_tensor.mean(dim=[1, 2])
            std += img_tensor.std(dim=[1, 2])
        mean /= len(self.img)
        std /= len(self.img)
        return mean, std
    
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, index):
        caption = self.caption[index]
        img_id  = f"{self.img[index]}.jpg"
        img = Image.open(self.img_dir/img_id)
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

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return images, captions    
    

dataset    = Build_Dataset(root_dir, img_dir, caption_file, True, img_size, augment=augment, split=None)
input_dim  = len(dataset.vocab)
pad_idx    = dataset.vocab.stoi["<pad>"]

def get_loaders():
    """Only called during training/evaluation, not inference."""
    train_dataset = Build_Dataset(root_dir, img_dir, caption_file, False, img_size, augment=augment, split="train")
    val_dataset   = Build_Dataset(root_dir, img_dir, caption_file, False, img_size, augment=False, split="val")
    test_dataset  = Build_Dataset(root_dir, img_dir, caption_file, False, img_size, augment=False, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=MyCollate(pad_idx), generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)

    return train_loader, val_loader, test_loader