import random

import torch

from src.constants.dataset import DatasetColumns
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class ReFLTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def __init__(
        self, *, min_mid_timestep: int, max_mid_timestep: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.min_mid_timestep = min_mid_timestep
        self.max_mid_timestep = max_mid_timestep

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
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        encoder_hidden_states = self.text_encoder(
            batch[DatasetColumns.tokenized_text.name]
        )[0]

        latents = torch.randn(
            (self.train_dataloader.batch_size, 4, 64, 64),
            device=self.device,
        )

        self.noise_scheduler.set_timesteps(self.max_mid_timestep, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        mid_timestep = random.randint(
            self.min_mid_timestep,
            self.max_mid_timestep - 1,
        )

        with torch.no_grad():
            for i, timestep in enumerate(timesteps[:mid_timestep]):
                latent_model_input = latents
                latent_model_input = self.noise_scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                noise_pred = self.latent_model(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                latents = self.noise_scheduler.step(
                    noise_pred, timestep, latents
                ).prev_sample

        latent_model_input = latents
        latent_model_input = self.noise_scheduler.scale_model_input(
            latent_model_input, timesteps[mid_timestep]
        )

        noise_pred = self.latent_model(
            latent_model_input,
            timesteps[mid_timestep],
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        pred_original_sample = self.noise_scheduler.step(
            noise_pred, timesteps[mid_timestep], latents
        ).pred_original_sample.to(self.weight_dtype)

        pred_original_sample /= self.vae.config.scaling_factor

        image = self.vae.decode(pred_original_sample).sample
        batch["image"] = image

        self.reward_model.score_grad(
            batch=batch,
            image=image,
        )

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
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
