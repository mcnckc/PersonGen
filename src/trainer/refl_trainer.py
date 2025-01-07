import random

import torch
from torch.cuda.amp import autocast

from src.constants.dataset import DatasetColumns
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class ReFLTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def process_batch(self, batch: dict[str, torch.Tensor], metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]

        with autocast():
            encoder_hidden_states = self.model.text_encoder(
                batch[DatasetColumns.tokenized_text.name]
            )[0]

            latents = torch.randn(
                (batch[DatasetColumns.tokenized_text.name].shape[0], 4, 64, 64),
                device=self.device,
            )

            self.model.noise_scheduler.set_timesteps(
                self.cfg_trainer.max_mid_timestep, device=self.device
            )
            timesteps = self.model.noise_scheduler.timesteps

            mid_timestep = (
                random.randint(
                    self.cfg_trainer.min_mid_timestep,
                    self.cfg_trainer.max_mid_timestep - 1,
                )
                if self.is_train
                else self.cfg_trainer.max_mid_timestep - 1
            )

            with torch.no_grad():
                for i, timestep in enumerate(timesteps[:mid_timestep]):
                    latent_model_input = latents
                    latent_model_input = self.model.noise_scheduler.scale_model_input(
                        latent_model_input, timestep
                    )
                    noise_pred = self.model.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                    latents = self.model.noise_scheduler.step(
                        noise_pred, timestep, latents
                    ).prev_sample

            latent_model_input = latents
            latent_model_input = self.model.noise_scheduler.scale_model_input(
                latent_model_input, timesteps[mid_timestep]
            )

            noise_pred = self.model.unet(
                latent_model_input,
                timesteps[mid_timestep],
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            pred_original_sample = self.model.noise_scheduler.step(
                noise_pred, timesteps[mid_timestep], latents
            ).pred_original_sample

            pred_original_sample /= self.model.vae.config.scaling_factor

            image = self.model.vae.decode(pred_original_sample).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor(image)

            batch["image"] = image

            if self.is_train:
                self.train_reward_model.score_grad(
                    batch=batch,
                    image=image,
                )
                batch["loss"] = batch["loss"] * self.cfg_trainer.loss_scale
                self.scaler.scale(batch["loss"]).backward()
            else:
                self.train_reward_model.score(
                    batch=batch,
                    image=image,
                )
                for reward_model in self.val_reward_models:
                    reward_model.score(batch=batch, image=image)
        # TODO: rewrite it!
        for loss_name in (
            self.loss_names_train if self.is_train else self.loss_names_test
        ):
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log generated images.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        self.writer.add_image(
            image_name=mode,
            image=batch["image"],
        )
