from pathlib import Path

import clip
import hydra
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from src.utils.io_utils import get_image_name_by_index


def get_tril_elements_mask(linear_size):
    mask = np.zeros((linear_size, linear_size), dtype=np.bool_)
    mask[np.tril_indices_from(mask)] = True
    np.fill_diagonal(mask, False)
    return mask


def _diversity_from_embeddings_pairwise_cosines(imgs_encoded: torch.Tensor):
    data = (imgs_encoded @ imgs_encoded.T).detach().cpu().numpy()
    mask = get_tril_elements_mask(data.shape[0])
    masked = data[mask].astype(np.float64)
    return masked


@hydra.main(version_base=None, config_path="../src/configs", config_name="lpips")
def main(config):
    for path in [
        "images_coco/lpips_image_refl_0.01_0",
        "images_coco/lpips_image_refl_0.01_20",
        "images_coco/lpips_image_refl_0.01_25",
        "images_coco/lpips_image_refl_0.01_32",
        "images_coco/lpips_image_refl_0.01_35",
        "images_coco/lpips_image_refl_0.01_38",
        "images_coco/lpips_refl_0",
        "images_coco/lpips_refl_20",
        "images_coco/lpips_refl_25",
        "images_coco/lpips_refl_32",
        "images_coco/lpips_refl_35",
        "images_coco/lpips_refl_38",
    ]:
        name = path.split("/")[-1]
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device
        images_path = Path(path)

        processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        model, transform = clip.load("ViT-B/32", device=device, jit=False)

        global_index = 0
        losses = []
        for _ in tqdm(range(config.prompts_count)):
            batch = [
                Image.open(images_path / get_image_name_by_index(global_index + i))
                for i in range(config.images_per_prompt)
            ]
            image_inputs = processor(
                images=batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).pixel_values.to(device)
            image_features = model.encode_image(image_inputs)
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
            masked = _diversity_from_embeddings_pairwise_cosines(image_features)
            losses.append(masked)
            global_index += config.images_per_prompt

        losses = np.array(losses)
        print(name)
        print(1 - losses.mean(), losses.std())


if __name__ == "__main__":
    main()
