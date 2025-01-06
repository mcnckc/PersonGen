import enum


class DatasetColumns(enum.Enum):
    tokenized_text = "tokenized_text"
    text_attention_mask = "text_attention_mask"
    image = "image"
