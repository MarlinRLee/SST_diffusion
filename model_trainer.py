from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from tqdm import tqdm
from losses import DiffusionLossCalculator

class DiffusionTrainer:
    def __init__(self, unet, scheduler, optimizer, log_dir="logs", num_prior=3, train_guidence="None", denoising_start = None):
        assert train_guidence in ["None", "Aux", "Partial"]
        self.unet = unet
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.num_prior = num_prior
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{train_guidence}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.loss_calculator = DiffusionLossCalculator(train_guidence)
        
        if denoising_start is None:
            denoising_start = scheduler.timesteps[0].item()
        self.denoising_start = denoising_start
        
    def prepare_data(self, batch, device):
        context_frames, mask, week_pred, future_offset, time_in_year = batch
        
        # Move tensors to device and convert to float32
        tensors = {
            "mask": mask.to(device, dtype=torch.float32),
            "week_pred": week_pred.to(device, dtype=torch.float32),
            "context_frames": context_frames.to(device, dtype=torch.float32),
            "future_offset": future_offset.to(device, dtype=torch.float32),
            "time_in_year": time_in_year.to(device, dtype=torch.float32)
        }
        
        # Split context and target frames
        target_frame = tensors["context_frames"][:, self.num_prior:, :, :]
        tensors["context_frames"] = tensors["context_frames"][:, :self.num_prior, :, :]
        
        # Get spatial dimensions from context frames
        batch_size, _, height, width = tensors["context_frames"].shape
        
        # Reshape time features to 4D (batch, channel, height, width)
        for key in ["future_offset", "time_in_year"]:
            feature = tensors[key]
            # Reshape from [batch] to [batch, 1, height, width]
            tensors[key] = feature.view(-1, 1, 1, 1).expand(-1, 1, height, width)
        
        return tensors, target_frame

    def train_step(self, batch, device):
        tensors, target_frame = self.prepare_data(batch, device)
        
        # Generate timesteps and noise
        timesteps = torch.randint(0, self.denoising_start, (target_frame.size(0),), device=device).long()
        noise = torch.randn_like(target_frame)
        noisy_target = self.scheduler.add_noise(target_frame, noise, timesteps)
        
        #print(tensors['future_offset'].shape)
        #print(tensors['time_in_year'].shape)
        #print(tensors['context_frames'].shape)
        #print(noisy_target.shape)
        
        # Model forward pass
        model_input = torch.cat([
            tensors['future_offset'],
            tensors['time_in_year'],
            tensors['context_frames'],
            noisy_target
        ], dim=1)
        
        outputs = self.unet(model_input, timesteps).sample
        prior_frame = tensors['context_frames'][:, self.num_prior - 1:self.num_prior, :, :]
        
        # Compute losses
        losses = self.loss_calculator.compute_losses(
            outputs, target_frame, prior_frame,
            tensors['week_pred'], tensors['mask'], timesteps
        )
        total_loss = self.loss_calculator.compute_total_loss(*losses)
        
        return total_loss, losses
    
    def train(self, train_dataloader, test_dataloader, num_epochs=5, device="cuda"):
        print("Start training", flush=True)
        self.unet.to(device, dtype=torch.float32)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            # Training loop
            self.unet.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    total_loss, losses = self.train_step(batch, device)
                
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                train_losses.append(losses[0].item())  # Store reconstruction loss
                
                # Log training metrics
                global_step = epoch * len(train_dataloader) + batch_idx
                self._log_metrics("train", losses, global_step)
            
            # Evaluation
            test_losses = self.evaluate(test_dataloader, device)
            if (epoch + 1) % 2 == 0:
                mae_metrics = self.evaluate_mae(test_dataloader, device)
                self.writer.add_scalar("test/mae", mae_metrics['mae'], epoch)
            
            # Log epoch metrics
            self._log_metrics("test", test_losses, epoch)
            print(f"Epoch {epoch + 1} - Train Loss: {sum(train_losses)/len(train_losses):.4f}, Test Loss: {test_losses[0]:.4f}")
        
        self.writer.close()

    def evaluate(self, dataloader, device):
            self.unet.eval()
            total_losses = [0.0] * 4  # For each type of loss
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    _, losses = self.train_step(batch, device)
                    for i, loss in enumerate(losses):
                        total_losses[i] += loss.item()
            
            return [loss / len(dataloader) for loss in total_losses]
        
    def generate(self, batch, device, num_inference_steps=50):
        """Generate predictions using the full diffusion process."""
        self.unet.eval()
        self.unet.to(device)
        
        with torch.no_grad():
            tensors, _ = self.prepare_data(batch, device)
            
            # Initialize latents from prior frame
            prior_frame = tensors['context_frames'][:, self.num_prior - 1:self.num_prior, :, :]
            latents = prior_frame.clone()
            
            # Set up diffusion timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in self.scheduler.timesteps:
                if t > self.denoising_start:
                    continue
                    
                timesteps = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
                
                # Prepare model input
                model_input = torch.cat([
                    tensors['future_offset'],
                    tensors['time_in_year'],
                    tensors['context_frames'],
                    latents
                ], dim=1)
                
                # Get model prediction and step scheduler
                noise_pred = self.unet(model_input, timesteps).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                # Apply mask
                latents = latents * tensors['mask']
                
            return latents
        
        
    def _log_metrics(self, stage, losses, step):
        """
        Log metrics to TensorBoard.
        
        Args:
            stage: str, either 'train' or 'test'
            losses: tuple of loss values
            step: int, the current step/epoch
        """
        loss_names = ['reconstruction', 'prior', 'energy', 'week_pred']
        
        for name, value in zip(loss_names, losses):
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"{stage}/{name}_loss", value, step)