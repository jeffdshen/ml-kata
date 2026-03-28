import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        super().__init__()
        self.inputs = torch.from_numpy(inputs).to(torch.float32)
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)
