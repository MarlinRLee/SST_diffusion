import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random

class SSTDataset(Dataset):
    def __init__(self, data_paths, mask_paths, week_pred_path, sequence_length = 3, min_future = 7, max_future = 14):#42-21
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.week_pred_path = week_pred_path
        self.sequence_length = sequence_length
        self.max_future = max_future
        self.min_future = min_future
        
        #self.data_files = [np.load(path, mmap_mode='r') for path in data_paths]
        #self.mask_files = [np.load(path, mmap_mode='r') for path in mask_paths]

        self.sst_data = np.stack([np.load(p) for p in data_paths], axis=0)#, mmap_mode='r'
        self.masks = np.stack([np.load(p) for p in mask_paths], axis=0)#, mmap_mode='r'
        self.week_pred_data = np.stack([np.load(p) for p in week_pred_path], axis=0)
        #assert np.allclose(self.sst_data, self.week_pred_data)
        self.time_length = self.sst_data.shape[1]
        
    def __len__(self):
        return self.sst_data.shape[0] * (self.time_length - self.sequence_length - self.max_future)

    def __getitem__(self, idx):
        file_idx = idx // (self.time_length - self.sequence_length - self.max_future)
        time_idx = idx % (self.time_length - self.sequence_length - self.max_future)
        sequence_end_idx = time_idx + self.sequence_length
        
        num_options = max(self.min_future, min(self.time_length - sequence_end_idx, self.max_future))
        future_offset = random.randint(self.min_future, num_options)
        
        frames = self.sst_data[file_idx, time_idx:sequence_end_idx]
        future_frame = self.sst_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1]
        frames = torch.cat((torch.tensor(frames), torch.tensor(future_frame)), dim=0) 
        
        week_pred = torch.tensor(self.week_pred_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1])
        
        #assert np.allclose(self.week_pred_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1], 
        #                   self.sst_data[file_idx, sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1])
        
        
        #assert torch.allclose(torch.tensor(future_frame), week_pred, atol=1e-7)
        #assert torch.equal(torch.tensor(future_frame), week_pred)
        
        zero_mask = frames == 0
        frames = (frames - 4.5899997) / (33.5 - 4.5899997)
        frames[zero_mask] = -1
        
        zero_mask = week_pred == 0
        week_pred = (week_pred - 4.5899997) / (33.5 - 4.5899997)
        week_pred[zero_mask] = -1
        
        mask = torch.tensor(self.masks[file_idx, sequence_end_idx + future_offset]).unsqueeze(0)
        future_offset_norm = future_offset / float(self.max_future)
        time_in_year = (time_idx % 365) / 365.0
        return frames, mask, week_pred, future_offset_norm, time_in_year

def prepare_datasets(data_paths, mask_paths, week_pred_path, sequence_length, batch_size=32, train_ratio=0.8):
    dataset = SSTDataset(data_paths, mask_paths, week_pred_path, sequence_length)
    
    # Calculate split indices for sequential splitting
    train_cutoff = int(train_ratio * dataset.time_length)
    
    # Create training and testing subsets based on time indices
    train_indices = []
    test_indices = []
    for file_idx in range(dataset.sst_data.shape[0]):  # Loop over all files

        train_indices += [file_idx * (dataset.time_length - dataset.sequence_length - dataset.max_future) + t
                          for t in range(train_cutoff - dataset.sequence_length - dataset.max_future)]

        test_indices += [file_idx * (dataset.time_length - dataset.sequence_length - dataset.max_future) + t
                         for t in range(train_cutoff, dataset.time_length - dataset.sequence_length - dataset.max_future)]

    # Create subsets using Subset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=2)
    
    return train_dataloader, test_dataloader