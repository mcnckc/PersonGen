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
        # 'lpips_refl_0',
        # 'lpips_refl_20',
        # 'lpips_refl_25',
        # 'lpips_refl_32',
        # 'lpips_refl_35',
        # 'lpips_refl_38',
        # 'lpips_image_refl_0.01_0',
        # 'lpips_image_refl_0.01_20',
        # 'lpips_image_refl_0.01_25',
        # 'lpips_image_refl_0.01_32',
        # 'lpips_image_refl_0.01_35',
        # 'lpips_image_refl_0.01_38',
        "fid_refl_coco_0/",
        "fid_refl_coco_20/",
        "fid_refl_coco_25/",
        "fid_refl_coco_32/",
        # "refl_coco_35/",
        # "refl_coco_38/",
        # 'lpips_refl_0.01_1_4_0',
        # 'lpips_refl_0.01_1_4_20',
        # 'lpips_refl_0.01_1_4_25',
        # 'lpips_refl_0.01_1_4_32',
        # 'lpips_refl_0.01_1_4_35',
        # 'lpips_refl_0.01_1_4_38',
    ],
    path_to_save="data/lpips/",
)
