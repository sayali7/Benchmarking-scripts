
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas as pd
import pickle


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def get_file_lists(meta_path, sample_column='SubID', phen_column='c15x'):
    # Load train/test split metadata
    with open(meta_path, 'rb') as handle:
        data = pickle.load(handle)

    # Map phenotype labels
    data['train'][phen_column] = data['train'][phen_column].map({"AD":1, "Control":0})
    data['test'][phen_column] = data['test'][phen_column].map({"AD":1, "Control":0})

    # Extract lists and label maps
    train_list = data['train'][sample_column].unique().tolist()
    train_label_map = dict(zip(data['train'][sample_column], data['train'][phen_column]))

    return train_list, train_label_map


class EmbDataset(Dataset):
    def __init__(self, file_list, data_path, label_mapping, cache_data=True, file_format="csv"):
        self.file_list = file_list
        self.data_path = data_path
        self.file_format = file_format
        self.label_mapping = label_mapping
        self.index_mapping = []
        self.sample_weights = []
        self.cache_data = cache_data
        self.file_cache = {}

        # Build index and weights
        for donor in file_list:
            file_path = os.path.join(data_path, donor + f".{self.file_format}")
            df = pd.read_csv(file_path, index_col=0) if self.file_format == "csv" else pd.read_parquet(file_path)
            n_cells = len(df)
            donor_weight = 1.0 / n_cells if n_cells > 0 else 0
            for i in range(n_cells):
                self.index_mapping.append((donor, i))
                self.sample_weights.append(donor_weight)
            if cache_data:
                self.file_cache[donor] = df.copy()
            del df

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        donor, local_idx = self.index_mapping[idx]
        if self.cache_data and donor in self.file_cache:
            df = self.file_cache[donor]
        else:
            file_path = os.path.join(self.data_path, donor + f".{self.file_format}")
            df = pd.read_csv(file_path, index_col=0) if self.file_format == "csv" else pd.read_parquet(file_path)
        row = df.iloc[local_idx]
        data = torch.tensor(row.values, dtype=torch.float32)
        label = torch.tensor(int(self.label_mapping.get(donor, -1)), dtype=torch.long)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return data, label, weight

