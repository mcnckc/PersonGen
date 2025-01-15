import random

import torch

from src.constants.dataset import DatasetColumns
from src.trainer.refl_trainer import ReFLTrainer


class ImageReFLTrainer(ReFLTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = 0

    def _sample_image_train(self, batch: dict[str, torch.Tensor]) -> None:
        if self.cfg_trainer.state_del is not None:
            self.state += 1
            self.state %= self.cfg_trainer.state_del

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

        noise = None

        with torch.no_grad():
            if (
                self.cfg_trainer.state_del is not None
                and self.state % self.cfg_trainer.state_del == 0
            ):
                latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                    latents=None,
                    start_timestamp_index=0,
                    end_timestamp_index=self.cfg_trainer.start_timestep_index,
                    batch=batch,
                    return_pred_original=False,
                )
            else:
                latents, noise = self.model.get_noisy_latents_from_images(
                    images=original_images,
                    timestep_index=self.cfg_trainer.start_timestep_index,
                )

        next_step_index = self.cfg_trainer.start_timestep_index
        if noise is not None:
            latents, noise_pred = self.model.predict_next_latents(
                latents=latents,
                timestep_index=next_step_index,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
                return_pred_original=False,
            )
            batch["loss"] = (
                torch.nn.functional.mse_loss(noise_pred, noise)
                * self.cfg_trainer.mse_loss_scale
            )
            batch["mse_loss"] = batch["loss"].detach().clone()
            next_step_index += 1
            latents = latents.detach()

        with torch.no_grad():
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=latents,
                start_timestamp_index=next_step_index,
                end_timestamp_index=mid_timestep,
                batch=batch,
                return_pred_original=False,
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestamp_index=mid_timestep,
            end_timestamp_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
        )
