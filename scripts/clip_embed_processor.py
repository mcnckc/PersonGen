from pathlib import Path

import clip
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipScore"


def main(device, paths, path_to_save, batch_size=512):
    with torch.no_grad():
        model, transform = clip.load("ViT-B/32", device=device, jit=False)
        path_to_save = Path(path_to_save)
        processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        for path in paths:
            directory = Path("images_coco/" + path)

            images = [
                Image.open(file) for file in directory.iterdir() if file.is_file()
            ]
            embeds = []
            for i in tqdm(range(0, len(images), batch_size)):
                batch = images[i : i + batch_size]
                image_inputs = processor(
                    images=batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).pixel_values.to(device)
                image_features = model.encode_image(image_inputs)
                image_features /= image_features.clone().norm(dim=-1, keepdim=True)
                embeds.append(image_features)
            embeds = torch.cat(embeds)
            torch.save(embeds, str(path_to_save / directory.name))


main(
    device="cuda:0",
    paths=[
        "image_refl_35_40_25",
        "image_refl_35_40_30",
        "image_refl_35_40_35",
        "image_refl_35_40_36",
        "image_refl_35_40_38",
        "image_refl_35_40_39",
    ],
    path_to_save="data/lpips/",
)
