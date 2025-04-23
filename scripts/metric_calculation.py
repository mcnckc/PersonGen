from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from torchvision import transforms
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn


def move_batch_to_device(batch, device):
    for column_name in batch:
        if isinstance(batch[column_name], torch.Tensor):
            batch[column_name] = batch[column_name].to(device)
    return batch


@hydra.main(
    version_base=None, config_path="../src/configs", config_name="metric_calculation"
)
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    reward_image_processor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    with torch.no_grad():
        reward_models = [
            instantiate(reward_model_cfg, device=device).to(device)
            for reward_model_cfg in config.reward_models
        ]
        reward_tables_mean = {}
        reward_tables_std = {}
        for reward_model in reward_models:
            data_path_mean = f"{reward_model.model_suffix}_mean.csv"
            data_path_std = f"{reward_model.model_suffix}_std.csv"

            reward_tables_mean[data_path_mean] = defaultdict(dict)
            reward_tables_std[data_path_std] = defaultdict(dict)

            if Path(data_path_mean).exists():
                reward_tables_mean[data_path_mean] |= pd.read_csv(
                    data_path_mean, index_col=0
                ).to_dict(orient="index")
            if Path(data_path_std).exists():
                reward_tables_std[data_path_std] |= pd.read_csv(
                    data_path_std, index_col=0
                ).to_dict(orient="index")

        for path in config.images_paths:
            step = int(path.split("_")[-1])
            dataset = instantiate(
                config.datasets.test,
                images_path=path,
                all_models_with_tokenizer=reward_models,
                dataset_split="test",
            )
            dataset.image_process = reward_image_processor
            dataloader = instantiate(
                config.dataloader.test,
                dataset=dataset,
                collate_fn=collate_fn,
            )
            for reward_model in reward_models:
                rewards = None
                for batch in tqdm(dataloader):
                    batch = move_batch_to_device(batch, device)
                    raw_images = batch["original_image"]
                    if rewards is None:
                        rewards = reward_model._get_reward(
                            batch=batch,
                            image=raw_images,
                        ).cpu()
                    else:
                        rewards = torch.cat(
                            [
                                rewards,
                                reward_model._get_reward(
                                    batch=batch,
                                    image=raw_images,
                                ).cpu(),
                            ]
                        )
                print(config.run_name, step)
                reward_tables_mean[f"{reward_model.model_suffix}_mean.csv"][step][
                    config.run_name
                ] = float(rewards.mean())
                reward_tables_std[f"{reward_model.model_suffix}_std.csv"][step][
                    config.run_name
                ] = float(rewards.std())

                for reward_tables in (reward_tables_mean, reward_tables_std):
                    for reward_table_path in reward_tables:
                        table = pd.DataFrame.from_dict(
                            reward_tables[reward_table_path], orient="index"
                        )
                        table.to_csv(reward_table_path)


if __name__ == "__main__":
    main()
