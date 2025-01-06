import typing as tp
from abc import abstractmethod

import torch.utils.checkpoint


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_suffix: str, reward_scale_factor: float, reward_offset: float
    ):
        super().__init__()
        self.model_suffix = model_suffix
        self.reward_scale_factor = reward_scale_factor
        self.reward_offset = reward_offset

    @abstractmethod
    def tokenize(
        self, batch: tp.Dict[str, tp.Any], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def score_grad(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> None:
        reward = self._get_reward(batch, image)
        loss = -(reward + self.reward_offset) * self.reward_scale_factor
        batch["loss"] = loss
        batch[self.model_suffix] = reward.mean().detach().item()

    def score(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            reward = self._get_reward(batch, image)
        batch[self.model_suffix] = reward.mean().detach().item()
