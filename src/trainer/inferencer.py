import torch
from tqdm.auto import tqdm

from src.constants.dataset import DatasetColumns
from src.metrics.tracker import MetricTracker
from src.models import StableDiffusion
from src.reward_models import BaseModel
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model: StableDiffusion,
        reward_models: list[BaseModel],
        config,
        device,
        dataloaders,
        writer,
        batch_transforms=None,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.writer = writer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.reward_models = reward_models

        # define metrics
        self.loss_names = [
            reward_model.model_suffix for reward_model in self.reward_models
        ]
        self.all_metrics = MetricTracker(
            *self.loss_names,
            *["improvement_" + model_suffix for model_suffix in self.loss_names],
            writer=self.writer,
        )
        self.start_timestep_index = None

        self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}

        for start_timestep_index in self.cfg_trainer.start_timestep_indexs:
            self.start_timestep_index = start_timestep_index
            for part, dataloader in self.evaluation_dataloaders.items():
                logs = self._inference_part(part, dataloader)
                part_logs[part] = logs

            self.writer.set_step(start_timestep_index)
            for metric_name in self.all_metrics.keys():
                self.writer.add_scalar(
                    f"{metric_name}", self.all_metrics.avg(metric_name)
                )
            self.all_metrics.reset()
        return part_logs

    def _sample_image(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )

        original_images = batch[DatasetColumns.original_image.name]

        noised_latents, _ = self.model.get_noisy_latents_from_images(
            images=original_images,
            timestep_index=self.start_timestep_index,
        )

        batch["image"] = self.model.sample_image(
            latents=noised_latents,
            start_timestep_index=self.start_timestep_index,
            end_timestep_index=self.cfg_trainer.end_timestep_index,
            batch=batch,
        )

    def process_batch(
        self, batch: dict[str, torch.Tensor], metrics: MetricTracker
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        self._sample_image(batch)

        original_image_batch = {
            **batch,
            "image": self.model.image_processor(
                batch[DatasetColumns.original_image.name]
            ),
        }

        for reward_model in self.reward_models:
            reward_model.score(batch=batch)
            reward_model.score(batch=original_image_batch)

        return batch, original_image_batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.all_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch, original_image_batch = self.process_batch(
                    batch=batch, metrics=self.all_metrics
                )

                for loss_name in self.loss_names:
                    if loss_name in batch:
                        self.all_metrics.update(loss_name, batch[loss_name].item())
                        self.all_metrics.update(
                            "improvement_" + loss_name,
                            batch[loss_name].item()
                            - original_image_batch[loss_name].item(),
                        )
        self.writer.add_image(
            image_name="generated",
            image=batch["image"],
        )
        self.writer.add_image(
            image_name="originals",
            image=original_image_batch["image"],
        )
        return self.all_metrics.result()
