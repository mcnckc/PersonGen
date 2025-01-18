import logging
import typing as tp

from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from src.constants.dataset import DatasetColumns

logger = logging.getLogger(__name__)


class DatasetWrapper(Dataset):
    """
    A wrapper for datasets that provides tokenization, optional image processing,
    and support for models with tokenizers.
    """

    def __init__(
        self,
        text_column: str,
        all_models_with_tokenizer: list,
        image_column: str | None = None,
        images_per_row: int | None = None,
        dataset_name: str | None = None,
        dataset_split: str | None = None,
        cache_dir: str | None = None,
        raw_dataset: Dataset | None = None,
        fixed_length: int | None = None,
    ):
        """
        Initializes the DatasetWrapper.

        Args:
            text_column (str): The name of the column containing text data.
            all_models_with_tokenizer (list): A list of models with tokenizers to apply to the text data.
            image_column (str | None): The name of the column containing image data, if applicable.
            images_per_row (int | None): Number of images per row for multi-image samples, if applicable.
            dataset_name (str | None): The name of the dataset to load (used if raw_dataset is not provided).
            dataset_split (str | None): The dataset split to load (used if raw_dataset is not provided).
            cache_dir (str | None): Directory to cache the dataset (used if raw_dataset is not provided).
            raw_dataset (datasets.Dataset | None): Preloaded raw dataset to wrap. If None, it will be loaded.
            fixed_length (int | None): Fixed length to override the actual dataset length.

        Raises:
            AssertionError: If the dataset does not contain required columns.
        """
        if raw_dataset is None:
            raw_dataset = datasets.load_dataset(
                dataset_name, split=dataset_split, cache_dir=cache_dir
            )
        if image_column is None:
            assert images_per_row is None

        self._assert_dataset_is_valid(
            raw_dataset=raw_dataset,
            text_column=text_column,
            image_column=image_column,
        )

        self.raw_dataset = raw_dataset
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.text_column = text_column
        self.image_column = image_column
        self.images_per_row = images_per_row
        self.fixed_length = fixed_length

        self.image_process = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, ind) -> dict[str, tp.Any]:
        """
        Fetches an item from the dataset.

        Args:
            ind (int): Index of the item to fetch.

        Returns:
            dict: A dictionary containing tokenized text and optionally processed image data.
        """
        if self.image_column is not None and self.images_per_row is not None:
            image_index = ind % self.images_per_row
            ind //= self.images_per_row

        data_dict = self.raw_dataset[ind]
        res = {}

        for model in self.all_models_with_tokenizer:
            res.update(
                model.tokenize(
                    data_dict[self.text_column],
                )
            )

        if self.image_column is not None:
            if self.images_per_row is not None:
                res[DatasetColumns.original_image.name] = self.image_process(
                    data_dict[self.image_column][image_index].convert("RGB")
                ).unsqueeze(0)
            else:
                res[DatasetColumns.original_image.name] = self.image_process(
                    data_dict[self.image_column].convert("RGB")
                ).unsqueeze(0)

        return res

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            int: The length of the dataset. If fixed_length is set, returns that value.
        """
        if self.fixed_length is not None:
            return self.fixed_length

        length = len(self.raw_dataset)
        if self.images_per_row is not None:
            return length * self.images_per_row
        return length

    @staticmethod
    def _assert_dataset_is_valid(
        raw_dataset: datasets.Dataset,
        text_column: str,
        image_column: str | None = None,
        images_per_row: int | None = None,
    ) -> None:
        """
        Validates the structure of the raw dataset.

        Args:
            raw_dataset (datasets.Dataset): The raw dataset to validate.
            text_column (str): The name of the text column that must be present in the dataset.
            image_column (str | None): The name of the image column, if applicable.

        Raises:
            AssertionError: If the required columns are not found in the dataset.
        """
        first_row = raw_dataset[0]

        assert (
            text_column in raw_dataset.column_names,
            "text_column must be present in raw_dataset",
        )
        assert isinstance(first_row[text_column], str)

        assert (
            image_column is None or image_column in raw_dataset.column_names,
            "image_column must be present in raw_dataset",
        )
        # TODO: add image validation
