import inspect
import typing as tp

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PixArtAlphaPipeline,
    StableDiffusionPipeline,
    Transformer2DModel,
)
from diffusers.image_processor import PixArtImageProcessor
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer

from src.constants.dataset import DatasetColumns
from src.models.base_model import BaseModel
from src.models.stable_diffusion import StableDiffusion


def retrieve_timesteps(
    scheduler,
    num_inference_steps: tp.Optional[int] = None,
    device: tp.Optional[tp.Union[str, torch.device]] = None,
    timesteps: tp.Optional[tp.List[int]] = None,
    sigmas: tp.Optional[tp.List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            DatasetColumns.tokenized_text.name: inputs.input_ids.squeeze(),
            DatasetColumns.text_attention_mask.name: inputs.attention_mask.squeeze(),
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch[DatasetColumns.text_attention_mask.name],
            timestep=timestep.int().expand(latents.shape[0]),
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

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def sample_image(
        self,
        latents: torch.Tensor | None,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prompt_embeds = batch[DatasetColumns.tokenized_text.name]
        prompt_attention_mask = batch[DatasetColumns.text_attention_mask.name]

        encoder_hidden_states = self.get_encoder_hidden_states(batch)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.noise_scheduler,
            end_timestep_index,
            prompt_embeds.device,
        )

        latent_channels = self.unet.config.in_channels
        latents = torch.randn(
            (encoder_hidden_states.shape[0], 4, 64, 64),
            device=encoder_hidden_states.device,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, 0.0)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop

        for i, t in enumerate(timesteps[:-1]):
            print(t)
            latent_model_input = latents
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor(
                    [current_timestep], dtype=dtype, device=latent_model_input.device
                )
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            noise_pred = self.unet(
                latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # learned sigma
            if self.unet.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            if num_inference_steps == 1:
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).pred_original_sample
            else:
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]

        image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        image = image_processor.resize_and_crop_tensor(image, 512, 512)
        image = image_processor.postprocess(image, output_type="pil")

        return image
