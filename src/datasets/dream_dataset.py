import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamDataset(Dataset):
    def __init__(self, target_prompts, all_models_with_tokenizer, inf_dataset=False):
        self.target_prompts = target_prompts
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.inf_dataset = inf_dataset
        self.len = 1000000000 if inf_dataset else len(self.target_prompts)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        res = {"prompt":self.target_prompts[0 if self.inf_dataset else idx]}
        for model in self.all_models_with_tokenizer:
            res.update(model.tokenize(res["prompt"]))
        return res
    