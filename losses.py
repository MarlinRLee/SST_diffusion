import torch
from torch import nn

class DiffusionLossCalculator:
    def __init__(self, guidence_type="None"):
        self.guidence_type = guidence_type
        self.loss_weights = self._initialize_loss_weights()
        
    def _initialize_loss_weights(self):
        if self.guidence_type == "None":
            return {
                "real": 1.0,
                "energy": 0.0,
                "prior": 0.0,
                "weak": 0.0
            }
        else:
            return {
                "real": 0.9,
                "energy": 0.2 * 0.002 * 6.25e-7,
                "prior": 0.2 * 0.002 * 500,
                "weak": 0.2 * 0.002 * 1000
            }
    
    def compute_energy(self, tensor):
        return torch.sum(tensor, dim=(1, 2, 3))
    
    def compute_losses(self, outputs, target_frame, prior_frame, week_pred, mask, timesteps):
        # Basic reconstruction loss
        recon_loss = (nn.MSELoss(reduction="none")(outputs, target_frame) * mask).mean()
        
        # Weekly prediction loss
        week_pred_loss = (nn.MSELoss(reduction="none")(outputs, week_pred) * mask)
        
        # Energy conservation loss
        energy_predicted = self.compute_energy(outputs * mask)
        energy_target = self.compute_energy(prior_frame * mask)
        energy_loss = nn.MSELoss(reduction="none")(energy_predicted, energy_target)
        
        # Prior frame prediction loss
        baseline_pred_loss = (nn.MSELoss(reduction="none")(prior_frame, outputs) * mask)
        
        # Apply partial guidance if needed
        if self.guidence_type == "Partial":
            week_pred_loss = week_pred_loss * (timesteps[:, None, None, None] >= 58).float()
            energy_loss = energy_loss * (timesteps[:, None] >= 162).float()
            baseline_pred_loss = baseline_pred_loss * (timesteps[:, None, None, None] >= 212).float()
            
        # Average losses
        week_pred_loss = week_pred_loss.mean()
        energy_loss = energy_loss.mean()
        baseline_pred_loss = baseline_pred_loss.mean()
        
        return recon_loss, baseline_pred_loss, energy_loss, week_pred_loss
    
    def compute_total_loss(self, *losses):
        recon_loss, baseline_pred_loss, energy_loss, week_pred_loss = losses
        return (
            self.loss_weights["real"] * recon_loss +
            self.loss_weights["prior"] * baseline_pred_loss +
            self.loss_weights["energy"] * energy_loss +
            self.loss_weights["weak"] * week_pred_loss
        )