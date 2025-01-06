import typing as tp

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.constants.dataset import DatasetColumns


class StableDiffusion(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name
        self.revision = revision

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=self.revision,
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.reward_model.requires_grad_(False)

    def tokenize(
        self, batch: tp.Dict[str, tp.Any], caption_column: str
    ) -> dict[str, tp.Any]:
        input_ids = self.tokenizer(
            batch[caption_column],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            DatasetColumns.tokenized_text.name: input_ids,
        }
