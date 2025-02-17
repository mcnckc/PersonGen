from pathlib import Path

import numpy as np
import torch


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


def main(paths, device, data_path, grouped_by, step=1):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    data_path = Path(data_path)
    for path in paths:
        name = path.split("/")[-1]
        file_path = data_path / name
        clip_embeds = torch.load(file_path).to(device)[::step]
        scores = np.array([])
        if grouped_by is None:
            scores = _diversity_from_embeddings_pairwise_cosines(clip_embeds)
        else:
            for i in range(0, clip_embeds.shape[0], grouped_by):
                scores = np.hstack(
                    (
                        scores,
                        _diversity_from_embeddings_pairwise_cosines(
                            clip_embeds[i : i + grouped_by]
                        ),
                    )
                )

        score = 1 - scores.mean()
        std = scores.std()
        mean_std = std / np.sqrt(scores.shape[0])
        print(name, score, std, mean_std)


if __name__ == "__main__":
    main(
        device="cuda:0",
        # paths=[
        #     'fid_refl_fid_0',
        #     'fid_refl_fid_20',
        #     'fid_refl_fid_25',
        #     'fid_refl_fid_30',
        #     'fid_refl_fid_32',
        #     'fid_refl_fid_35',
        #     'fid_refl_fid_38',
        #     'fid_refl_0.01_0',
        #     'fid_refl_0.01_20',
        #     'fid_refl_0.01_25',
        #     'fid_refl_0.01_32',
        #     'fid_refl_0.01_35',
        #     'fid_refl_0.01_36',
        #     'fid_refl_0.01_37',
        #     'fid_refl_0.01_38',
        #     'images_coco/lpips_refl_0.01_1_4_0',
        #     'images_coco/lpips_refl_0.01_1_4_20',
        #     'images_coco/lpips_refl_0.01_1_4_25',
        #     'images_coco/lpips_refl_0.01_1_4_32',
        #     'images_coco/lpips_refl_0.01_1_4_35',
        #     'images_coco/lpips_refl_0.01_1_4_38',
        # ],
        paths=[
            # "refl_coco_0",
            # "refl_coco_20",
            # "refl_coco_25",
            # "refl_coco_32",
            # "refl_coco_35",
            # "refl_coco_38",
            "fid_refl_coco_0",
            "fid_refl_coco_20",
            "fid_refl_coco_25",
            "fid_refl_coco_32",
        ],
        data_path="data/lpips/",
        grouped_by=None,
    )
