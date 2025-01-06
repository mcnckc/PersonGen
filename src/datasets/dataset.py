import logging

from torch.utils.data import Dataset

import datasets
from src.constants.dataset import DatasetColumns

logger = logging.getLogger(__name__)


class DatasetWrapper(Dataset):
    def __init__(
        self,
        raw_dataset: datasets.Dataset,
        all_models_with_tokenizer: list,
        text_column: str,
        image_column: str | None = None,
    ):
        self._assert_dataset_is_valid(
            raw_dataset=raw_dataset,
            text_column=text_column,
            image_column=image_column,
        )
        self.raw_dataset = raw_dataset
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.text_column = text_column
        self.image_column = image_column

    def __getitem__(self, ind):
        data_dict = self.raw_dataset[ind]

        res = {}

        for model in self.all_models_with_tokenizer:
            res.update(
                model.tokenize(
                    data_dict[self.text_column],
                    caption_column=self.text_column,
                )
            )
        if self.image_column is not None:
            res[DatasetColumns.image.name] = data_dict[self.image_column]
        return res

    def __len__(self):
        return len(self.raw_dataset)

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
