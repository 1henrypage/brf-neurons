
import torch
from numpy import load
from torch.utils.data import TensorDataset, DataLoader


def shd_to_dataset(
        input_file_path: str,
        label_file_path: str,
) -> TensorDataset:

    inputs = load(input_file_path)
    inputs = torch.Tensor(inputs)

    targets = load(label_file_path).astype(float)
    targets = torch.Tensor(targets).long()

    return TensorDataset(inputs, targets)

