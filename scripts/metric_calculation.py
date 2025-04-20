import hydra
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

        for path in [
            "/home/jovyan/shares/SR006.nfs2/torchrik/refl_proj/HumanDiffusion/images_coco/image_refl_xl_v_25",
        ]:
            dataset = instantiate(
                config.datasets.test,
                images_path=path,
                all_models_with_tokenizer=reward_models,
                dataset_split="test",
            )
            dataloader = instantiate(
                config.dataloader.test,
                dataset=dataset,
                collate_fn=collate_fn,
            )
            rewards = None
            for reward_model in reward_models:
                for batch in tqdm(dataloader):
                    batch = move_batch_to_device(batch, device)
                    raw_images = (batch["original_image"] / 2 + 0.5).clamp(0, 1)

                    if rewards is None:
                        rewards = reward_model._get_reward(
                            batch=batch,
                            image=raw_images,
                        ).cpu()
                        print(rewards)
                    else:
                        rewards = torch.cat(
                            [
                                rewards,
                                reward_model._get_reward(
                                    batch=batch,
                                    image=reward_image_processor(raw_images),
                                ).cpu(),
                            ]
                        )
                print(reward_model.model_suffix, rewards.mean(), rewards.std())


if __name__ == "__main__":
    main()
