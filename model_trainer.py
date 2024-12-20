from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from tqdm import tqdm
from losses import DiffusionLossCalculator

class DiffusionTrainer:
    def __init__(self, unet, scheduler, optimizer, 
                 log_dir="logs", 
                 num_prior=3, 
                 train_guidence = "Aux", 
                 denoising_start = None,
                 lr_scheduler=None):
        assert train_guidence in ["None", "Aux", "Partial"]
        self.unet = unet
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.num_prior = num_prior
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{train_guidence}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.loss_calculator = DiffusionLossCalculator(train_guidence)
        self.lr_scheduler =lr_scheduler
        
        if denoising_start is None:
            denoising_start = scheduler.timesteps[0].item()
        self.denoising_start = denoising_start
        
    def prepare_data(self, batch, device):
        context_frames, mask, week_pred, time_in_year = batch
        
        # Move tensors to device and convert to float32
        tensors = {
            "mask": mask.to(device, dtype=torch.float32),
            "week_pred": week_pred.to(device, dtype=torch.float32),
            "context_frames": context_frames.to(device, dtype=torch.float32),
            "time_in_year": time_in_year.to(device, dtype=torch.float32)
        }
        
        # Split context and target frames
        target_frame = tensors["context_frames"][:, self.num_prior:, :, :]
        tensors["context_frames"] = tensors["context_frames"][:, :self.num_prior, :, :]
        
        # Get spatial dimensions from context frames
        batch_size, _, height, width = tensors["context_frames"].shape
        
        # Reshape time features to 4D (batch, channel, height, width)
        feature = tensors["time_in_year"]
        # Reshape from [batch] to [batch, 1, height, width]
        tensors["time_in_year"] = feature.view(-1, 1, 1, 1).expand(-1, 1, height, width)
    
        return tensors, target_frame

    def train_step(self, batch, device):
        tensors, target_frame = self.prepare_data(batch, device)
        
        # Generate timesteps and noise
        timesteps = torch.randint(0, self.denoising_start, (target_frame.size(0),), device=device).long()
        noise = torch.randn_like(target_frame)
        noisy_target = self.scheduler.add_noise(target_frame, noise, timesteps)
        
        
        # Model forward pass
        model_input = torch.cat([
            tensors['time_in_year'],
            tensors['context_frames'],
            noisy_target
        ], dim=1)
        
        outputs = self.unet(model_input, timesteps).sample
        prior_frame = tensors['context_frames'][:, self.num_prior - 1:self.num_prior, :, :]
        
        # Compute losses
        losses = self.loss_calculator.compute_losses(
            outputs, target_frame, prior_frame,
            tensors['week_pred'], tensors['mask']
        )
        total_loss = self.loss_calculator.compute_total_loss(*losses, timesteps = timesteps)
        
        recon_loss, baseline_pred_loss, energy_loss, week_pred_loss = losses
        week_pred_loss = week_pred_loss.mean()
        energy_loss = energy_loss.mean()
        baseline_pred_loss = baseline_pred_loss.mean()
        losses = recon_loss, baseline_pred_loss, energy_loss, week_pred_loss
        
        return total_loss, losses
    
    def train(self, train_dataloader, test_dataloader, num_epochs=5, device="cuda"):
        print("Start training", flush=True)
        self.unet.to(device, dtype=torch.float32)
        scaler = torch.cuda.amp.GradScaler()
        self.unet.train()
        for epoch in range(num_epochs):
            # Training loop
            train_losses = []
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    total_loss, losses = self.train_step(batch, device)
                
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                self.lr_scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("train/learning_rate", current_lr, epoch * len(train_dataloader) + batch_idx)
                
                
                train_losses.append(losses[0].item())  # Store reconstruction loss
                
                # Log training metrics
                global_step = epoch * len(train_dataloader) + batch_idx
                self._log_metrics("train", losses, global_step)
                
                if (batch_idx + 1) % 50 == 0:
                    self.unet.eval()
                    train_mae = self.evaluate_mae(train_dataloader, device, short = True)
                    self.writer.add_scalar("train/mae", train_mae, global_step)
                    self.unet.train()
                    
            self.unet.eval()
            # Evaluation
            test_losses = self.evaluate(test_dataloader, device)
            #if (epoch + 1) % 2 == 0:
            test_mae = self.evaluate_mae(test_dataloader, device)
            self.writer.add_scalar("test/mae", test_mae, epoch)
            # Log epoch metrics
            self._log_metrics("test", test_losses, epoch)
            print(f"Epoch {epoch + 1} - Train Loss: {sum(train_losses)/len(train_losses):.4f}, Test Loss: {test_losses[0]:.4f}")
            self.unet.train()
        
        self.writer.close()

    def evaluate(self, dataloader, device):
            total_losses = [0.0] * 4  # For each type of loss
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    _, losses = self.train_step(batch, device)
                    for i, loss in enumerate(losses):
                        total_losses[i] += loss.item()
            
            return [loss / len(dataloader) for loss in total_losses]
        
        
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
            
    def evaluate_mae(self, dataloader, device, num_inference_steps = 100, short = False):
        """
        Evaluate the model using Mean Absolute Error on the full generation process.
        """
        total_masked_mae = 0.0
        num_batches = 0
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating MAE")):
                
                tensors, target_frame = self.prepare_data(batch, device)
                
                # Initialize latents from prior frame
                prior_frame = tensors['context_frames'][:, self.num_prior - 1:self.num_prior, :, :]
                latents = prior_frame.clone()
                
                # Add noise to match the noise level of the first timestep
                noise = torch.randn_like(latents)
                timesteps = torch.full((latents.shape[0],), self.denoising_start, device=device, dtype=torch.long)
                latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # Denoising loop
                for t in self.scheduler.timesteps:
                    if t > self.denoising_start:
                        continue
                        
                    timesteps = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
                    
                    # Prepare model input
                    model_input = torch.cat([
                        tensors['time_in_year'],
                        tensors['context_frames'],
                        latents
                    ], dim=1)
                    
                    model_output = self.unet(model_input, timesteps).sample
                
                    latents = self.scheduler.step(
                        model_output,
                        t,
                        latents,
                    ).prev_sample
                    
                    # Apply mask
                    latents = latents * tensors['mask']
                
                # Calculate MAE
                mae = torch.abs(latents - target_frame)
                masked_mae = mae * tensors['mask']
                
                # Compute batch metrics
                batch_mae = mae.mean().item()
                batch_masked_mae = (masked_mae.sum() / tensors['mask'].sum()).item()
                
                # Aggregate metrics
                total_masked_mae += batch_masked_mae
                num_batches += 1
                if short:
                    break


        # Calculate final metrics
        final_masked_mae = total_masked_mae / num_batches

        print(f"Final Masked MAE: {final_masked_mae:.4f}")

        return final_masked_mae
        
