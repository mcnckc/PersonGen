import logging

from torch.utils.data import Dataset

import datasets
from datasets import load_dataset
from src.constants.dataset import DatasetColumns

logger = logging.getLogger(__name__)


class DatasetWrapper(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_split: str,
        is_pair_dataset: bool,
        cache_dir: str,
        text_column: str,
        all_models_with_tokenizer: list,
        image_column: str | None = None,
    ):
        raw_dataset = load_dataset(
            dataset_name, split=dataset_split, cache_dir=cache_dir
        )
        self._assert_dataset_is_valid(
            raw_dataset=raw_dataset,
            text_column=text_column,
            image_column=image_column,
        )
        self.raw_dataset = raw_dataset
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.text_column = text_column
        self.image_column = image_column
        self.is_pair_dataset = is_pair_dataset

    def __getitem__(self, ind):
        if self.is_pair_dataset and self.image_column is not None:
            pair_index = ind % 2
            ind //= 2

        data_dict = self.raw_dataset[ind]
        res = {}

        for model in self.all_models_with_tokenizer:
            res.update(
                model.tokenize(
                    data_dict[self.text_column],
                )
            )

        if self.image_column is not None:
            if self.is_pair_dataset:
                res[DatasetColumns.image.name] = data_dict[self.image_column][
                    pair_index
                ]
            else:
                res[DatasetColumns.image.name] = data_dict[self.image_column]

        return res

    def __len__(self):
        length = len(self.raw_dataset)
        if self.is_pair_dataset and self.image_column is not None:
            return length * 2
        return length

    @staticmethod
    def _assert_dataset_is_valid(
        raw_dataset: datasets.Dataset,
        text_column: str,
        image_column: str | None = None,
    ) -> None:
        """
        Check is raw dataset contains all required fields.
        """
        assert (
            text_column in raw_dataset.column_names,
            "text_column must be present in raw_dataset",
        )

        assert (
            image_column is None or image_column in raw_dataset.column_names,
            "image_column must be present in raw_dataset",
        )
