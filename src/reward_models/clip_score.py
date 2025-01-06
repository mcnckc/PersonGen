import typing as tp

import clip
import torch
from torchvision import transforms

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipScore"


class ClipScore(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, transform = clip.load("ViT-B/32", device=device, jit=False)
        self.model = model
        self.transform = transform
        self.device = device
        self.image_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def tokenize(
        self, batch: tp.Dict[str, tp.Any], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        caption = batch[caption_column]
        processed_caption = clip.tokenize(caption)

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        tokenized_caption = batch[
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"
        ]
        processed_image = self.image_processor(image)
        candidates = self.model.encode_text(tokenized_caption)
        images = self.model.encode_image(processed_image)

        images = torch.nn.functional.normalize(images)
        candidates = torch.nn.functional.normalize(candidates)

        reward = torch.clip(torch.sum(images * candidates, axis=1), 0, None)
        return reward
