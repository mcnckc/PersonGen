import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    for column_name in dataset_items[0].keys():
        result_batch[column_name] = torch.vstack(
            [elem[column_name] for elem in dataset_items]
        )

    return result_batch
