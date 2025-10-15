import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamDataset(Dataset):
    def __init__(self, target_prompt, all_models_with_tokenizer):
        self.target_prompt = target_prompt
        self.all_models_with_tokenizer = all_models_with_tokenizer

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        res = {"prompt":self.target_prompt}
        for model in self.all_models_with_tokenizer:
            res.update(model.tokenize(self.target_prompt))
        return res
    