from src.trainer import (
    DraftKTrainer,
    ImageReFLTrainer,
    ImageReFLTrainerV2,
    Inferencer,
    InferencerV2,
    InferencerV3,
    ReFLTrainer,
)

TRAINER_NAME_TO_CLASS = {
    "DraftK": DraftKTrainer,
    "ReFL": ReFLTrainer,
    "ImageReFL": ImageReFLTrainer,
    "ImageReFLV2": ImageReFLTrainerV2,
}

INFERENCER_NAME_TO_CLASS = {
    "Inference": Inferencer,
    "InferenceV2": InferencerV2,
    "InferenceV3": InferencerV3,
}
