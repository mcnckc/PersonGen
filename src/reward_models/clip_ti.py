import typing as tp

import clip
import torch
import os
from PIL import Image
from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel
from torchvision import transforms

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipTI"


class ClipTI(BaseModel):
    def __init__(self, device: torch.device, clip_prompt: str, src_img_dir: str):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, transform = clip.load(MODEL_NAME, device=device, jit=False)
        self.device = device
        self.model = model
        self.clip_resolution = 224
        self.tensor_preproc = transforms.Compose(
            [
                transforms.Resize(
                    self.clip_resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(self.clip_resolution),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        print('VIT32 transform', transform)
        with torch.no_grad():
            self.tg_prompt = self.model.encode_text(clip.tokenize(
                clip_prompt,
                truncate=True,
            ).to(device))
            self.tg_prompt = torch.nn.functional.normalize(self.tg_prompt, dim=-1)

            src_image_paths = []
            src_images = []
            valid_extensions = {'.jpg', '.jpeg', '.png'}
            for filename in os.listdir(src_img_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    src_image_paths.append(os.path.join(src_img_dir, filename))
            if not src_image_paths:
                raise ValueError(f"Directory {src_img_dir} has no images")
            for img_path in src_image_paths:
                image = Image.open(img_path)
                src_images.append(transform(image))
            src_images = torch.stack(src_images, dim=0).to(device)
            self.src_images = self.model.encode_image(src_images)

            self.src_images = torch.nn.functional.normalize(self.src_images, dim=-1)
        

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = clip.tokenize(
            caption,
            truncate=True,
        )

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        tg_image = self.model.encode_image(self.tensor_preproc(image))
        tg_image = torch.nn.functional.normalize(tg_image, dim=-1)
        clipT = self.tg_prompt @ tg_image.T
        clipI = tg_image @ self.src_images.T
        clipT = clipT.mean()
        clipI = clipI.mean()
        batch["clip_t"] = clipT
        batch["clip_i"] = clipI
        return clipT * clipI
