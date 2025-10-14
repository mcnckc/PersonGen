import typing as tp

import clip
import torch

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel
from torchvision import transforms

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipTI"


class ClipTI(BaseModel):
    def __init__(self, device: torch.device, clip_prompt: str):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, transform = clip.load(MODEL_NAME, device=device, jit=False)
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
        self.tg_prompt = self.model.encode_text(clip.tokenize(
            clip_prompt,
            truncate=True,
        ).to(device))
        print(self.tg_prompt.requires_grad)
        self.tg_prompt.requires_grad_(False)
        self.tg_prompt = torch.nn.functional.normalize(self.tg_prompt, dim=-1)
        self.device = device

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
        src_images = self.model.encode_image(self.tensor_preproc(batch["src_images"]))
        tg_image = torch.nn.functional.normalize(tg_image, dim=-1)
        src_images = torch.nn.functional.normalize(src_images, dim=-1)
        clipT = self.tg_prompt @ tg_image.T
        clipI = tg_image @ src_images.T
        clipT = clipT.mean()
        clipI = clipI.mean()
        batch["clip_t"] = clipT
        batch["clip_i"] = clipI
        return clipT * clipI
