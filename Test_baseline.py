import torch
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np

def normalize_data(train_data, full_data):
    # Create a mask for zeros (previously NaN)
    zero_mask = full_data == 0
    # Compute min and max from the training data, ignoring NaN
    non_zero_train_data = train_data[train_data != 0]
    data_min = torch.min(non_zero_train_data)
    data_max = torch.max(non_zero_train_data)

    # Normalize the full dataset using training min-max
    normalized_data = (full_data - data_min) / (data_max - data_min)
    
    # Clip values to [0, 1] in case of rounding issues
    normalized_data = torch.clip(normalized_data, 0, 1)

    # Set NaN values to -1
    normalized_data[zero_mask] = -1

    return normalized_data

class SSTDataset(Dataset):
    def __init__(self, sst_data, masks, sequence_length=6, max_future = 21):
        
        self.sst_data = sst_data
        self.masks = masks
        self.sequence_length = sequence_length
        self.max_future = max_future

    def __len__(self):
        return self.sst_data.size(0) - self.sequence_length - self.max_future

    def __getitem__(self, idx):
        sequence_end_idx = idx + self.sequence_length
        
        frames = self.sst_data[idx:sequence_end_idx]
        
        num_options = max(1, min(self.sst_data.size(0) - (sequence_end_idx), self.max_future))
        future_offset = random.randint(1, num_options)
        
        future_frame = self.sst_data[sequence_end_idx + future_offset:sequence_end_idx + future_offset + 1]
        frames = torch.cat((frames, future_frame), dim=0)
        
        mask = self.masks[sequence_end_idx + future_offset].unsqueeze(0)

        return frames, mask, future_offset / float(self.max_future), (idx % 365 / 365.0)
    
    
from diffusers import DDPMScheduler
print("Start", flush=True)
sst_data = np.load('data/processed_sst_data.npy')  # Shape: (time, lat, lon)
masks = np.load('data/sst_masks.npy')  # Shape: (time, lat, lon)
print(sst_data.shape, flush = True)

# Ensure the data and masks match
assert sst_data.shape == masks.shape, "Data and masks must have the same shape."

sst_data_tensor = torch.tensor(sst_data, dtype=torch.float32)
masks_tensor = torch.tensor(masks, dtype=torch.float32)

# Train-test split
train_size = int(0.8 * len(sst_data_tensor))
train_data_tensor = sst_data_tensor[:train_size]


# Normalize the data
sst_data_normalized = normalize_data(train_data_tensor, sst_data_tensor)
train_data_tensor = sst_data_normalized[:train_size]
test_data_tensor = sst_data_normalized[train_size:]

print("Make data loader")
# Dataset and DataLoader
sequence_length = 14
test_dataset = SSTDataset(test_data_tensor, masks_tensor, sequence_length=sequence_length)

test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers = 4)
scheduler = DDPMScheduler(num_train_timesteps=1000)


from torch import nn
import numpy as np
from tqdm import tqdm

def evaluate(dataloader, scheduler, num_prior, device):
        total_loss = 0.0
        for context_frames, mask, _, _ in dataloader:
                mask = mask.to(device, dtype=torch.float32)
                
                context_frames = context_frames.to(device, dtype=torch.float32)
                target_frame = context_frames[:, num_prior:, :, :]
                context_frames = context_frames[:, num_prior - 1:num_prior, :, :]
                
                # Add noise to target latent
                noise = torch.randn_like(target_frame)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (target_frame.size(0),), device=device)
                noisy_latent_target = scheduler.add_noise(target_frame, noise, timesteps)
                
                predicted_frame = context_frames
                
                alpha_bar = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                predicted_epsilon = (noisy_latent_target - alpha_bar.sqrt() * predicted_frame) / (1 - alpha_bar).sqrt()

                # Compute loss
                loss = (nn.MSELoss(reduction="none")(predicted_epsilon, noise) * mask).mean()
                total_loss += loss.item()

        return total_loss / len(dataloader)



print("eval baseline")
# Evaluate the persistence model
device = "cuda" if torch.cuda.is_available() else "cpu"
persistence_loss = evaluate(test_dataloader, scheduler, 14, device)
print(f"Persistence Model Test Loss: {persistence_loss:.4f}", flush=True)