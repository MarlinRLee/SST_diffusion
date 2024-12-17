import torch
from model_trainer import DiffusionTrainer
from Data_Loader import prepare_datasets 
from diffusers import UNet2DModel, DDPMScheduler#AutoencoderKL

def main():
    print("Load Data", flush=True)
    num_files = 6#3
    data_path_template = 'data/processed_sst_data{}.npy'
    mask_path_template = 'data/sst_masks{}.npy'
    good_pred_template = 'data/good_pred{}.npy'

    data_paths = [data_path_template.format(i) for i in range(num_files)]
    mask_paths = [mask_path_template.format(i) for i in range(num_files)]
    good_pred_paths = [good_pred_template.format(i) for i in range(num_files)]

    sequence_length = 3
    train_dataloader, test_dataloader = prepare_datasets(
        data_paths, mask_paths, good_pred_paths, train_ratio=0.9, sequence_length = sequence_length, batch_size=32
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
    trainer = DiffusionTrainer(unet, scheduler, optimizer, num_prior=sequence_length, train_guidence = train_guidence)

    print("Train Model", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.train(train_dataloader, test_dataloader, num_epochs=10, device=device)

if __name__ == "__main__":
    print("Start", flush=True)
    main()