import torch
import pandas as pd

from PIL import Image

class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, input_filename, transform=None, target_transform=None, img_key='filepath', target_key='label', sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.targets = df[target_key].tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(self.targets[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
