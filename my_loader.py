from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


class dataloader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        df = pd.read_csv(self.data_path, names=[0, 1, 2, 3, 4])
        d = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        data = np.array(df.iloc[:, 0:4])
        label = df.iloc[:, 4].map(d)

        self.data = torch.from_numpy(
            np.array(
                (data - np.mean(data, axis=0)) / np.std(data, axis=0), dtype="float32"
            )
        )
        self.label = torch.from_numpy(np.array(label, dtype="int64"))

        self.length = len(label)
        print(f"读出数据{self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.label[index]
