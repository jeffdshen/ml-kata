from torch.utils.data import Dataset
import torch

class NumpyDataset(Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs = torch.from_numpy(inputs).to(torch.float32)
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
    def __len__(self):
        return len(self.targets)
