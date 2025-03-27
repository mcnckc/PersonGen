from src.trainer import (
    CombinedGenerationInferencer,
    DraftKTrainer,
    ImageReFLTrainer,
    ImageReFLTrainerV2,
    Inferencer,
    InferencerV3,
    ProFusionInferencer,
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
    "CombinedGenerationInferencer": CombinedGenerationInferencer,
    "ProFusionInferencer": ProFusionInferencer,
    "InferenceV3": InferencerV3,
}

REQUIRES_ORIGINAL_MODEL = [
    "CombinedGenerationInferencer",
    "ProFusionInferencer",
    "InferenceV3",
]
