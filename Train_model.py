import torch
from model_trainer import DiffusionTrainer
from Data_Loader import prepare_datasets 
from diffusers import UNet2DModel, DDPMScheduler#AutoencoderKL
import numpy as np
def find_threshold(train_dataloader, scheduler):
    Threshold_distance = 10.177961
    # Initialize Accumulators
    total_alpha_context_target = 0.0
    num_batches = 0

    # Training Loop
    for context_frames, _, _, _, _ in train_dataloader:
        # Batch-wise processing
        target_frame = context_frames[:, -1:, :, :]
        context_frame = context_frames[:, :1, :, :]


        # Compute signal differences
        signal_diff_context = torch.norm(context_frame - target_frame, dim=(2, 3))[:, 0].mean().item()
        
        alpha_context_target = 0
        alpha_values = np.linspace(0, 1, 100001)
        for alpha in reversed(alpha_values):
            #if np.sqrt(alpha) * signal_diff <= 1.96 * np.sqrt(2 * (1 - alpha)):
            if alpha * signal_diff_context <= Threshold_distance * np.sqrt(2 * (1 - alpha)):
                alpha_context_target = alpha
                break
            
        # Update accumulators
        total_alpha_context_target += alpha_context_target
        num_batches += 1
    # Compute averages
    average_alpha_context_target = total_alpha_context_target / num_batches
    
    alphas_cumprod = scheduler.alphas_cumprod.numpy()
    
    return np.argmin(np.abs(alphas_cumprod - average_alpha_context_target))


def main():
    print("Load Data", flush=True)
    num_files = 1#3
    data_path_template = 'data/processed_sst_data{}.npy'
    mask_path_template = 'data/sst_masks{}.npy'
    good_pred_template = 'data/good_pred{}.npy'

    data_paths = [data_path_template.format(i) for i in range(num_files)]
    mask_paths = [mask_path_template.format(i) for i in range(num_files)]
    good_pred_paths = [good_pred_template.format(i) for i in range(num_files)]

    sequence_length = 3
    train_dataloader, test_dataloader = prepare_datasets(
        data_paths, mask_paths, good_pred_paths, train_ratio=0.9, sequence_length = sequence_length, batch_size=8
    )

    print("Load Model", flush=True)
    unet = UNet2DModel(
        sample_size=64,
        in_channels=sequence_length + 3,
        out_channels=1,
        layers_per_block=1,
        dropout=0.6
    )
    
    print("Load trainer", flush=True)
    train_guidence = "None"
    print(train_guidence, flush=True)
    scheduler = DDPMScheduler(num_train_timesteps = 1000, prediction_type="sample")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=1e-4)
    
    start_step = find_threshold(train_dataloader, scheduler)
    
    trainer = DiffusionTrainer(unet, scheduler, optimizer, num_prior=sequence_length, train_guidence = train_guidence, denoising_start = start_step)

    print("Train Model", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.train(train_dataloader, test_dataloader, num_epochs=10, device=device)

if __name__ == "__main__":
    print("Start", flush=True)
    main()