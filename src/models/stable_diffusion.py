import typing as tp

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from src.constants.dataset import DatasetColumns
from src.models.base_model import BaseModel


class StableDiffusion(BaseModel):
    """
    A Stable Diffusion model wrapper that provides functionality for text-to-image synthesis,
    noise scheduling, latent space manipulation, and image decoding.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
    ) -> None:
        """
        Initializes the components of  StableDiffusion model.

        Args:
            pretrained_model_name (str): The name or path of the pretrained model to use.
            revision (str | None): The specific revision of the model to use. Defaults to None.

        Notes:
            The Variational Autoencoder (VAE) and text encoder parameters are frozen to prevent gradient updates.
        """
        super().__init__(pretrained_model_name=pretrained_model_name, revision=revision)

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
        """
        Sets the U-Net model to training mode.

        Args:
            mode (bool): Whether to enable or disable training mode.
        """
        self.unet.train()

    def eval(self, mode: bool = True):
        """
        Sets the U-Net model to evaluation mode.

        Args:
            mode (bool): Whether to enable or disable evaluation mode.
        """
        self.unet.eval()

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        """
        Tokenizes the given caption.

        Args:
            caption (str): The text to tokenize.

        Returns:
            dict: A dictionary containing the tokenized text tensor.
        """
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
        """
        Sets the timesteps for the noise scheduler.

        Args:
            max_timestep (int): The maximum timestep for the diffusion process.
            device (torch.device): The device to set the timesteps on.
        """
        self.noise_scheduler.set_timesteps(max_timestep, device=device)
        self.timesteps = self.noise_scheduler.timesteps

    def get_noisy_latents_from_images(
        self, images: torch.Tensor, timestep_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the latents obtained from the input images.

        Args:
            images (torch.Tensor): Input image tensor.
            timestep_index (int): Index of the timestep for adding noise.

        Returns:
            tuple: A tuple containing the noisy latents and the noise tensor.
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        noisy_images = self.noise_scheduler.add_noise(
            original_samples=latents,
            noise=noise,
            timesteps=(
                torch.ones((images.shape[0],), device=latents.device).int()
                * self.timesteps[timestep_index]
            ),
        )
        return noisy_images, noise

    def get_encoder_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Retrieves the hidden states from the text encoder.

        Args:
            batch (dict): A batch containing tokenized text.

        Returns:
            torch.Tensor: Hidden states from the text encoder.
        """
        return self.text_encoder(
            batch[DatasetColumns.tokenized_text.name],
        )[0]

    def predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        batch: dict[str, torch.Tensor],
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the next latent states during the diffusion process.

        Args:
            latents (torch.Tensor): Current latent states.
            timestep_index (int): Index of the current timestep.
            encoder_hidden_states (torch.Tensor): Encoder hidden states from the text encoder.
            batch (dict): Input batch containing data.
            return_pred_original (bool): Whether to return the predicted original sample.

        Returns:
            tuple: Next latents and predicted noise tensor.
        """
        timestep = self.timesteps[timestep_index]

        latent_model_input = self.noise_scheduler.scale_model_input(
            sample=latents, timestep=timestep
        )

        noise_pred = self.unet(
            latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        sample = self.noise_scheduler.step(
            model_output=noise_pred, timestep=timestep, sample=latents
        )

        latents = (
            sample.pred_original_sample if return_pred_original else sample.prev_sample
        )

        return latents, noise_pred

    def do_k_diffusion_steps(
        self,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multiple diffusion steps between specified timesteps.

        Args:
            start_timestep_index (int): Starting timestep index.
            end_timestep_index (int): Ending timestep index.
            batch (dict): Input batch containing data.
            latents (torch.Tensor | None): Initial latents (optional).
            encoder_hidden_states (torch.Tensor | None): Encoder hidden states (optional).
            return_pred_original (bool): Whether to return the predicted original sample.

        Returns:
            tuple: Resulting latents and encoder hidden states.
        """
        assert start_timestep_index <= end_timestep_index

        if encoder_hidden_states is None:
            encoder_hidden_states = self.get_encoder_hidden_states(batch)

        if latents is None:
            latents = torch.randn(
                (encoder_hidden_states.shape[0], 4, 64, 64),
                device=encoder_hidden_states.device,
            )

        for timestep_index in range(start_timestep_index, end_timestep_index - 1):
            latents, _ = self.predict_next_latents(
                latents=latents,
                timestep_index=timestep_index,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
                return_pred_original=False,
            )
        res, _ = self.predict_next_latents(
            latents=latents,
            timestep_index=end_timestep_index - 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            return_pred_original=return_pred_original,
        )
        return res, encoder_hidden_states

    def sample_image(
        self,
        latents: torch.Tensor | None,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generates an image sample by decoding the latents.

        Args:
            latents (torch.Tensor | None): Initial latents (optional).
            start_timestep_index (int): Starting timestep index.
            end_timestep_index (int): Ending timestep index.
            batch (dict): Input batch containing data.
            encoder_hidden_states (torch.Tensor | None): Encoder hidden states (optional).

        Returns:
            torch.Tensor: Decoded image sample.
        """
        pred_original_sample, _ = self.do_k_diffusion_steps(
            latents=latents,
            start_timestep_index=start_timestep_index,
            end_timestep_index=end_timestep_index,
            batch=batch,
            encoder_hidden_states=encoder_hidden_states,
            return_pred_original=True,
        )

        pred_original_sample /= self.vae.config.scaling_factor

        image = self.vae.decode(pred_original_sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return self.image_processor(image)
