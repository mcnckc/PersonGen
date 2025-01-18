import typing as tp

import torch
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer

from src.constants.dataset import DatasetColumns
from src.models.base_model import BaseModel
from src.models.stable_diffusion import StableDiffusion


class PixArt(StableDiffusion):
    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
    ) -> None:
        BaseModel.__init__(
            self, pretrained_model_name=pretrained_model_name, revision=revision
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.timesteps = self.noise_scheduler.timesteps

        self.text_encoder = T5EncoderModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision
        )

        self.unet = Transformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
        )

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            DatasetColumns.tokenized_text.name: inputs.input_ids,
            DatasetColumns.text_attention_mask.name: inputs.attention_mask,
        }

    def get_encoder_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.text_encoder(
            batch[DatasetColumns.tokenized_text.name],
            batch[DatasetColumns.text_attention_mask.name],
        )[0]

    def predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        batch: dict[str, torch.Tensor],
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self.timesteps[timestep_index]
        latent_model_input = self.noise_scheduler.scale_model_input(latents, timestep)
        noise_pred = self.unet(
            latent_model_input,
            encoder_attention_mask=batch[DatasetColumns.text_attention_mask.name],
            timestep=torch.ones((latents.shape[0],), device=latents.device).int()
            * self.timesteps[timestep_index],
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        ).sample.chunk(2, 1)[0]

        if return_pred_original:
            pred_original_sample = self.noise_scheduler.step(
                noise_pred, timestep, latents
            ).pred_original_sample

            pred_original_sample /= self.vae.config.scaling_factor
            return pred_original_sample, noise_pred

        latents = self.noise_scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents, noise_pred
