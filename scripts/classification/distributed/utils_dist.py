import pandas as pd
import tqdm
import pickle
import os

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.distributed as dist

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
# Initialize the distributed environment.
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'  # Use any free port.
    #os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    # Use NCCL backend for GPUs; use "gloo" for CPU training.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Clean up the distributed environment.
def cleanup():
    dist.destroy_process_group()
    print(f"Rank {os.environ['RANK']} cleaned up")

def load_data(train_test_split, data_path, sample_column='SubID'):
    with open(f'{train_test_split}', 'rb') as handle:
        data = pickle.load(handle)

    train_meta = data['train']
    test_meta = data['test']

    print ("loading train data...")
    train_data = pd.DataFrame()
    for file in tqdm.tqdm(train_meta[sample_column].unique()[:]):
        df = pd.read_csv(data_path+"/"+file+".csv", index_col=0)
        train_data = pd.concat([train_data, df])

    print ("loading test data...")
    test_data = pd.DataFrame()
    for file in tqdm.tqdm(test_meta[sample_column].unique()[:]):
        df = pd.read_csv(data_path+"/"+file+".csv", index_col=0)
        test_data = pd.concat([test_data, df])

    return train_data, test_data, train_meta, test_meta



########################### Try lazy data-loading #######################


def get_file_lists(train_test_meta, sample_column='SubID', phen_column = "c15x",):
    with open(f'{train_test_meta}', 'rb') as handle:
        data = pickle.load(handle)

    data['train'][phen_column] = data['train'][phen_column].map({"AD":1,"Control":0})
    data['test'][phen_column] = data['test'][phen_column].map({"AD":1,"Control":0})

    train_label_map = dict(zip(data['train'][sample_column], data['train'][phen_column]))
    test_label_map = dict(zip(data['train'][sample_column], data['train'][phen_column]))

    train_list = []
    test_list = []

    for file in tqdm.tqdm(data['train'][sample_column].unique()[:]):
        train_list.append(file)

    
    for file in tqdm.tqdm(data['train'][sample_column].unique()[:]):
        test_list.append(file)

    return train_list, test_list, train_label_map, test_label_map

def get_sharded_file_list(full_file_list, rank, world_size):
    n_files = len(full_file_list)
    files_per_proc = n_files // world_size
    remainder = n_files % world_size

    if rank < remainder:
        start = rank * (files_per_proc + 1)
        end = start + files_per_proc + 1
        start=0
        end=100
    else:
        start = rank * files_per_proc + remainder
        end = start + files_per_proc
        start=100
        end=200

    return full_file_list[start:end]

class EmbDataset(Dataset):
    def __init__(self, file_list, data_path, label_mapping, cache_data=True, file_format="csv"):
        """
        Parameters:
          file_list (list): List of donor (file) basenames assigned to this process.
          data_path (str): Directory where the files are stored.
          label_mapping (dict): Mapping from donor (file basename) to label.
          file_format (str): 'parquet' or 'csv'
        """
        self.file_list = file_list
        self.data_path = data_path
        self.file_format = file_format
        self.label_mapping = label_mapping  # donor -> label
        self.index_mapping = []   # List of tuples: (donor, row index)
        self.sample_weights = []  # Weight for each sample
        self.cache_data = cache_data
        self.file_cache = {}

        # Build the index and compute weights.
        for donor in file_list:
            file_path = os.path.join(data_path, donor + f".{file_format}")
            # Load file only to count cells.
            if file_format == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, index_col=0)
            n_cells = len(df)
            # Compute weight: each cell gets weight = 1/n_cells so that donor’s total weight = 1.
            donor_weight = 1.0 / n_cells if n_cells > 0 else 0
            for i in range(n_cells):
                self.index_mapping.append((donor, i))
                self.sample_weights.append(donor_weight)

            if self.cache_data:
                self.file_cache[donor] = df.copy()
                
            del df

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        donor, local_idx = self.index_mapping[idx]
        # If caching is enabled and data is already loaded, re-use it.
        if self.cache_data and donor in self.file_cache:
            df = self.file_cache[donor]
        else:
            file_path = os.path.join(self.data_path, donor + f".{self.file_format}")
            if self.file_format == "csv":
                df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.read_parquet(file_path)
        row = df.iloc[local_idx]
        # Convert row data to tensor. Adjust indexing if needed.
        data = torch.tensor(row.values, dtype=torch.float32)
        label_value = self.label_mapping.get(donor, -1)
        label = torch.tensor(int(label_value), dtype=torch.long)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return data, label, weight

