import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamDataset(Dataset):
    def __init__(self, img_dir, target_prompt, all_models_with_tokenizer, transform=None, target_size=(512, 512)):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.target_prompt = target_prompt
        self.all_models_with_tokenizer = all_models_with_tokenizer
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])
        
        self.image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for filename in os.listdir(img_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                self.image_paths.append(os.path.join(img_dir, filename))
        
        if not self.image_paths:
            raise ValueError(f"Directory {img_dir} has no images")


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        res = {"prompt":self.target_prompt, "images":[]}
        for model in self.all_models_with_tokenizer:
            res.update(model.tokenize(self.target_prompt))
        img_path = self.image_paths[idx]
        
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            res["images"].append(image)
        res["src_images"] = torch.stack(res["images"], dim=0)
        return res
    