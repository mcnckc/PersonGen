import typing as tp

import clip
import torch

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipTI"


class ClipTI(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, transform = clip.load(MODEL_NAME, device=device, jit=False)
        self.model = model
        self.transform = transform
        print('VIT32 transform', self.transform)
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
        tg_prompt = self.model.encode_text(batch[
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"
        ])
        print('Shapes:')
        print(image, image.shape)
        print(batch["src_images"], len(batch["src_images"]), batch["src_images"][0].shape)
        tg_image = self.model.encode_image(self.transform(image))
        print("TG EMBEDDING SHAPE", tg_image.shape)
        src_images = self.model.encode_image(self.transform(batch["src_images"]))
        print("SRC batch SHAPE", src_image.shape)
        tg_prompt = torch.nn.functional.normalize(tg_prompt, dim=-1)
        tg_image = torch.nn.functional.normalize(tg_image, dim=-1)
        src_images = torch.nn.functional.normalize(src_images, dim=-1)

        clipT = tg_prompt @ tg_image.T
        clipI = tg_image @ src_images.T
        print(f"CLIP SHAPES: T:{clipT} I:{clipI}")
        clipT = clipT.mean()
        clipI = clipI.mean()
        return clipT * clipI
