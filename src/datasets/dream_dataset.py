import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamDataset(Dataset):
    def __init__(self, target_prompt, all_models_with_tokenizer, inf_dataset=False):
        self.target_prompt = target_prompt
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.len = 1000000000 if inf_dataset else 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        res = {"prompt":self.target_prompt}
        for model in self.all_models_with_tokenizer:
            res.update(model.tokenize(self.target_prompt))
        return res
    