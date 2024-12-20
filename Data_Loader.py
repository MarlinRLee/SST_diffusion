import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random

class SSTDataset(Dataset):
    def __init__(self, data_paths, mask_paths, week_pred_path, sequence_length = 3):#42-21
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.week_pred_path = week_pred_path
        self.sequence_length = sequence_length
        self.future_pred = 7
        
        #self.data_files = [np.load(path, mmap_mode='r') for path in data_paths]
        #self.mask_files = [np.load(path, mmap_mode='r') for path in mask_paths]

        self.sst_data = np.stack([np.load(p) for p in data_paths], axis=0)#, mmap_mode='r'
        self.masks = np.stack([np.load(p) for p in mask_paths], axis=0)#, mmap_mode='r'
        self.week_pred_data = np.stack([np.load(p) for p in week_pred_path], axis=0)
        #assert np.allclose(self.sst_data, self.week_pred_data)
        self.time_length = self.sst_data.shape[1]
        
    def __len__(self):
        return self.sst_data.shape[0] * (self.time_length - self.sequence_length - self.future_pred)

    def __getitem__(self, idx):
        file_idx = idx // (self.time_length - self.sequence_length - self.future_pred)
        time_idx = idx % (self.time_length - self.sequence_length - self.future_pred)
        sequence_end_idx = time_idx + self.sequence_length
        
        future_offset = 7
        
        frames = self.sst_data[file_idx, time_idx:sequence_end_idx]
        future_frame = self.sst_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1]
        frames = torch.cat((torch.tensor(frames), torch.tensor(future_frame)), dim=0) 
        
        week_pred = torch.tensor(self.week_pred_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1])
        
        zero_mask = frames == 0
        frames = (frames - 4.5899997) / (33.5 - 4.5899997)
        frames[zero_mask] = -1
        
        zero_mask = week_pred == 0
        week_pred = (week_pred - 4.5899997) / (33.5 - 4.5899997)
        week_pred[zero_mask] = -1
        
        mask = torch.tensor(self.masks[file_idx, sequence_end_idx + future_offset]).unsqueeze(0)
        time_in_year = (time_idx % 365) / 365.0
        
        return frames, mask, week_pred, time_in_year

def prepare_datasets(data_paths, mask_paths, week_pred_path, sequence_length, batch_size=32, train_ratio=0.8):
    dataset = SSTDataset(data_paths, mask_paths, week_pred_path, sequence_length)
    
    # Calculate split indices for sequential splitting
    train_cutoff = int(train_ratio * dataset.time_length)
    
    # Create training and testing subsets based on time indices
    train_indices = []
    test_indices = []
    random.seed(42)
    for file_idx in range(dataset.sst_data.shape[0]):  # Loop over all files

        train_idx  = [file_idx * (dataset.time_length - dataset.sequence_length - dataset.future_pred) + t
                          for t in range(train_cutoff - dataset.sequence_length - dataset.future_pred)]

        #train_idx = random.sample(train_idx, 3 * len(train_idx) // 4)
        train_indices.extend(train_idx)

        test_idx = [file_idx * (dataset.time_length - dataset.sequence_length - dataset.future_pred) + t
                   for t in range(train_cutoff, dataset.time_length - dataset.sequence_length - dataset.future_pred)]
        
        #test_idx = random.sample(test_idx, 3 * len(test_idx) // 4)
        test_indices.extend(test_idx)

    # Create subsets using Subset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, 
                                  num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, 
                                 num_workers=2)
    
    return train_dataloader, test_dataloader