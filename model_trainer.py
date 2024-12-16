from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
from datetime import datetime

def compute_energy(tensor):
    # Compute total thermal energy for the batch
    return torch.sum(tensor, dim=(1, 2, 3))

class DiffusionTrainer:
    def __init__(self, unet, scheduler, optimizer, log_dir="logs", num_prior = 3, train_guidence = "None"):
        assert train_guidence in ["None", "Aux", "Partial"]
        self.unet = unet
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{train_guidence}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir = log_dir
        self.num_prior = num_prior
        self.train_guidence = train_guidence

    def prepare_data(self, context_frames, mask, week_pred, future_offset, time_in_year, device):
        mask = mask.to(device, dtype=torch.float32)
        week_pred = week_pred.to(device, dtype=torch.float32)
        context_frames = context_frames.to(device, dtype=torch.float32)
        target_frame = context_frames[:, self.num_prior:, :, :]
        context_frames = context_frames[:, :self.num_prior, :, :]
        
        # Expand dims for offsets and time
        future_offset = future_offset.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        time_in_year = time_in_year.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dim = context_frames.size(2)
        
        future_offset = future_offset.repeat(1, 1, dim, dim)
        time_in_year = time_in_year.repeat(1, 1, dim, dim)
        
        return context_frames, target_frame, mask, week_pred, future_offset, time_in_year

    def compute_loss(self, context_frames, target_frame, mask, week_pred, future_offset, time_in_year, device):
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (target_frame.size(0),), device=device).long()
        noise = torch.randn_like(target_frame)
        noisy_latent_target = self.scheduler.add_noise(target_frame, noise, timesteps)
        
        input_val = torch.cat([future_offset, time_in_year, context_frames, noisy_latent_target], dim=1)
        outputs = self.unet(input_val, timesteps).sample
        
        loss = (nn.MSELoss(reduction="none")(outputs, target_frame) * mask).mean()
        
        week_pred_loss = (nn.MSELoss(reduction="none")(outputs, week_pred) * mask)
        
        
        # Energy loss
        prior_frame = context_frames[:, self.num_prior - 1:self.num_prior, :, :]
        
        energy_predicted = compute_energy(outputs * mask)
        energy_target = compute_energy(prior_frame * mask)
        energy_loss = nn.MSELoss(reduction="none")(energy_predicted, energy_target)
        
        baseline_pred_loss = (nn.MSELoss(reduction="none")(prior_frame, outputs) * mask)
        
        if self.train_guidence == "Partial":
            week_pred_loss = week_pred_loss * (timesteps[:, None, None, None] >= 58).float()
            energy_loss = energy_loss * (timesteps[:, None] >= 162).float()
            baseline_pred_loss = baseline_pred_loss * (timesteps[:, None, None, None] >= 212).float()
        week_pred_loss = week_pred_loss.mean()
        energy_loss = energy_loss.mean()
        baseline_pred_loss = baseline_pred_loss.mean()
            
        return loss, baseline_pred_loss, energy_loss, week_pred_loss

    def train(self, train_dataloader, test_dataloader, num_epochs=5, device="cuda"):
        if self.train_guidence == "None":
            energy_alpha = 0.0
            prior_alpha = 0.0
            weak_alpha = 0.0
            real_alpha = 1.0
        else: 
            energy_alpha = .2 *.002 * 6.25e-7#s
            prior_alpha = .2 *.002 * 500
            weak_alpha = .2 *.002 * 1000
            real_alpha = .9
        print("Start training", flush=True)
        self.unet.to(device, dtype=torch.float32)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train loop
            self.unet.train()
            train_loss = 0.0
            for batch_idx, (context_frames, mask, week_pred, future_offset, time_in_year) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                context_frames, target_frame, mask, week_pred, future_offset, time_in_year = self.prepare_data(
                    context_frames, mask, week_pred, future_offset, time_in_year, device
                )
                
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, baseline_pred_loss, energy_loss, week_pred_loss = self.compute_loss(
                        context_frames, target_frame, mask, week_pred, future_offset, time_in_year, device
                    )

                # Total loss includes energy loss
                tot_loss = real_alpha * loss + prior_alpha * baseline_pred_loss + energy_alpha * energy_loss + weak_alpha * week_pred_loss
                
                # Backpropagation
                scaler.scale(tot_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()
                global_step = epoch * len(train_dataloader) + batch_idx
                self.writer.add_scalar("train/real", loss.item(), global_step)
                self.writer.add_scalar("train/prior",  baseline_pred_loss.item(), global_step)
                self.writer.add_scalar("train/energy",  energy_loss.item(), global_step)
                self.writer.add_scalar("train/week_pred",  week_pred_loss.item(), global_step)
                #test_loss

            train_loss /= len(train_dataloader)
            print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            # Test loop
            test_loss, baseline_pred_loss, energy_loss, week_pred_loss = self.evaluate(test_dataloader, device)
            print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}")

            # Log test loss
            self.writer.add_scalar("test/real", loss, epoch)
            self.writer.add_scalar("test/prior",  baseline_pred_loss, epoch)
            self.writer.add_scalar("test/energy",  energy_loss, epoch)
            self.writer.add_scalar("test/week_pred",  week_pred_loss, epoch)

        self.writer.close()    
    
    def generate(self, context_frames, mask, week_pred, future_offset, time_in_year, device, num_inference_steps=50):
        """
        Generate predictions using the full diffusion process.
        
        Args:
            context_frames: Input context frames tensor
            mask: Mask tensor for valid regions
            week_pred: Weekly prediction tensor
            future_offset: Future offset tensor
            time_in_year: Time in year tensor
            device: Device to run generation on
            num_inference_steps: Number of denoising steps
        
        Returns:
            Tensor of generated predictions
        """
        self.unet.eval()
        
        with torch.no_grad():
            # Prepare input data
            context_frames, _, mask, week_pred, future_offset, time_in_year = self.prepare_data(
                context_frames, mask, week_pred, future_offset, time_in_year, device
            )
            
            # Initialize noise
            latents = torch.randn(
                (context_frames.shape[0], context_frames.shape[1] - self.num_prior, 
                context_frames.shape[2], context_frames.shape[3]),
                device=device
            )
            
            # Set number of inference steps
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in self.scheduler.timesteps:
                # Expand timestep tensor
                timesteps = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
                
                # Prepare model input
                model_input = torch.cat([future_offset, time_in_year, context_frames, latents], dim=1)
                
                # Get model prediction
                noise_pred = self.unet(model_input, timesteps).sample
                
                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                latents = latents * mask
                
            return latents

    def evaluate(self, dataloader, device, num_inference_steps=50):
        """
        Evaluate the model using Mean Absolute Error on the full generation process.
        
        Args:
            dataloader: DataLoader containing evaluation data
            device: Device to run evaluation on
            num_inference_steps: Number of denoising steps
        
        Returns:
            Dictionary containing MAE metrics
        """
        self.unet.eval()
        total_mae = 0.0
        total_masked_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for context_frames, mask, week_pred, future_offset, time_in_year in tqdm(dataloader, desc="Evaluating MAE"):
                # Generate predictions
                predictions = self.generate(
                    context_frames, mask, week_pred, future_offset, time_in_year, 
                    device, num_inference_steps
                )
                
                # Get target frames
                target_frame = context_frames[:, self.num_prior:, :, :].to(device)
                mask = mask.to(device)
                
                # Calculate MAE
                mae = torch.abs(predictions - target_frame)
                masked_mae = mae * mask
                
                # Aggregate metrics
                total_mae += mae.mean().item()
                total_masked_mae += (masked_mae.sum() / mask.sum()).item()
                num_batches += 1
        
        return {
            'mae': total_mae / num_batches,
            'masked_mae': total_masked_mae / num_batches
        }
        
        
        
        """    def evaluate(self, dataloader, device):
        self.unet.eval()
        total_loss = 0.0
        total_baseline_pred_loss = 0.0
        total_energy_loss = 0.0
        total_week_pred_loss = 0
        with torch.no_grad():
            for context_frames, mask, week_pred, future_offset, time_in_year in tqdm(dataloader):
                context_frames, target_frame, mask, week_pred, future_offset, time_in_year = self.prepare_data(
                    context_frames, mask, week_pred, future_offset, time_in_year, device
                )
                loss, baseline_pred_loss, energy_loss, week_pred_loss = self.compute_loss(
                    context_frames, target_frame, mask, week_pred, future_offset, time_in_year, device
                )
                total_loss += loss.item()
                total_baseline_pred_loss += baseline_pred_loss.item()
                total_energy_loss += energy_loss.item()
                total_week_pred_loss += week_pred_loss.item()

        return total_loss / len(dataloader), total_baseline_pred_loss / len(dataloader), total_energy_loss / len(dataloader), total_week_pred_loss / len(dataloader)
    """