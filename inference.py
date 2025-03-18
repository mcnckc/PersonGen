import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.constants.trainer import INFERENCER_NAME_TO_CLASS
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_v2")
def main(config):
    set_random_seed(config.inferencer.seed)
    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, None, project_config)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    model = instantiate(config.model).to(device)
    reward_models = [
        instantiate(reward_model_cfg, device=device).to(device)
        for reward_model_cfg in config.reward_models
    ]
    all_models_with_tokenizer = reward_models + [model]

    dataloaders, batch_transforms = get_dataloaders(
        config, device, all_models_with_tokenizer
    )

    original_model = None

    if config.inferencer.type not in INFERENCER_NAME_TO_CLASS:
        raise ValueError(f"Inference type must be one of {INFERENCER_NAME_TO_CLASS}")

    inferencer_cls = INFERENCER_NAME_TO_CLASS[config.inferencer.type]

    if config.inferencer.type in ("InferenceV2", "InferenceV3"):
        original_model = instantiate(config.model).to(device)

    inferencer = inferencer_cls(
        model=model,
        reward_models=reward_models,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        writer=writer,
        original_model=original_model,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
