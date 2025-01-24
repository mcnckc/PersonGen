from src.trainer import (
    DraftKTrainer,
    ImageReFLTrainer,
    Inferencer,
    InferencerV2,
    ReFLTrainer,
)

TRAINER_NAME_TO_CLASS = {
    "DraftK": DraftKTrainer,
    "ReFL": ReFLTrainer,
    "ImageReFL": ImageReFLTrainer,
}

INFERENCER_NAME_TO_CLASS = {
    "Inference": Inferencer,
    "InferenceV2": InferencerV2,
}
