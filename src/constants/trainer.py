from src.trainer import DraftKTrainer, ImageReFLTrainer, ReFLTrainer

TRAINER_NAME_TO_CLASS = {
    "DraftK": DraftKTrainer,
    "ReFL": ReFLTrainer,
    "ImageReFL": ImageReFLTrainer,
}
