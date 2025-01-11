import random

import torch

from src.constants.dataset import DatasetColumns
from src.trainer.refl_trainer import ReFLTrainer


class ImageReFLTrainer(ReFLTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def _sample_image_train(self, batch: dict[str, torch.Tensor]) -> None:
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep + 1,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )
        original_images = batch[DatasetColumns.original_image.name]

        with torch.no_grad():
            original_latents, noise = self.model.get_noisy_latents_from_images(
                images=original_images,
                timestep_index=self.cfg_trainer.start_timestep_index,
            )

            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=original_latents,
                start_timestamp_index=self.cfg_trainer.start_timestep_index,
                end_timestamp_index=mid_timestep,
                input_ids=batch[DatasetColumns.tokenized_text.name],
                return_pred_original=False,
            )

        if self.cfg_trainer.with_original_loss and self.is_train:
            noise_pred = self.model.unet(
                original_latents,
                self.cfg_trainer.start_timestep_index,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            batch["loss"] = (
                torch.nn.functional.mse_loss(noise_pred, noise)
                * self.cfg_trainer.mse_loss_scale
            )

            batch["mse_loss"] = batch["loss"].detach().clone()

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestamp_index=mid_timestep,
            end_timestamp_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
        )
