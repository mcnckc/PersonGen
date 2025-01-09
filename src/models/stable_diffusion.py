import typing as tp

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from torchvision import transforms
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
        self.timesteps = self.noise_scheduler.timesteps

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

    def train(self, mode: bool = True):
        self.unet.train()

    def eval(self, mode: bool = True):
        self.unet.eval()

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            DatasetColumns.tokenized_text.name: input_ids,
        }

    def set_timesteps(self, max_timestep: int, device) -> None:
        self.noise_scheduler.set_timesteps(max_timestep, device=device)
        self.timesteps = self.noise_scheduler.timesteps

    def get_noisy_latents_from_images(
        self, images: torch.Tensor, timestep_index: int
    ) -> torch.Tensor:
        latents = self.vae.encode(images).latent_dist.sample()

        noise = torch.randn_like(latents)
        noisy_images = self.noise_scheduler.add_noise(
            latents,
            noise,
            timesteps=torch.ones((images.shape[0],), device=latents.device).int()
            * self.timesteps[timestep_index],
        )
        return noisy_images

    def _predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        return_pred_original: bool = False,
    ) -> torch.Tensor:
        timestep = self.timesteps[timestep_index]
        latent_model_input = self.noise_scheduler.scale_model_input(latents, timestep)
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        if return_pred_original:
            pred_original_sample = self.noise_scheduler.step(
                noise_pred, timestep, latents
            ).pred_original_sample

            pred_original_sample /= self.vae.config.scaling_factor
            return pred_original_sample

        latents = self.noise_scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents

    def do_k_diffusion_steps(
        self,
        start_timestamp_index: int,
        end_timestamp_index: int,
        latents: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert start_timestamp_index < end_timestamp_index
        if encoder_hidden_states is None:
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        if latents is None:
            latents = torch.randn(
                (encoder_hidden_states.shape[0], 4, 64, 64),
                device=encoder_hidden_states.device,
            )

        for timestep_index in range(start_timestamp_index, end_timestamp_index - 1):
            latents = self._predict_next_latents(
                latents=latents,
                timestep_index=timestep_index,
                encoder_hidden_states=encoder_hidden_states,
                return_pred_original=False,
            )
        res = self._predict_next_latents(
            latents=latents,
            timestep_index=end_timestamp_index - 1,
            encoder_hidden_states=encoder_hidden_states,
            return_pred_original=return_pred_original,
        )
        return res, encoder_hidden_states

    def sample_image(
        self,
        latents: torch.Tensor,
        start_timestamp_index: int,
        end_timestamp_index: int,
        input_ids: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_original_sample, _ = self.do_k_diffusion_steps(
            latents=latents,
            start_timestamp_index=start_timestamp_index,
            end_timestamp_index=end_timestamp_index,
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_pred_original=True,
        )
        image = self.vae.decode(pred_original_sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return self.image_processor(image)
