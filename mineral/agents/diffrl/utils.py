import numpy as np
import torch


def adaptive_scheduler(current_lr, kl_dist, kl_threshold=0.008, min_lr=1e-6, max_lr=1e-2, decay=1.5):
    lr = current_lr
    if kl_dist > (2.0 * kl_threshold):
        lr = max(current_lr / decay, min_lr)
    if kl_dist < (0.5 * kl_threshold):
        lr = min(current_lr * decay, max_lr)
    return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """For normal distribution."""
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    # return kl.mean()
    return kl


def soft_update(module, module_target, alpha: float):
    for param, param_targ in zip(module.parameters(), module_target.parameters()):
        param_targ.data.mul_(alpha)
        param_targ.data.add_((1.0 - alpha) * param.data)


def grad_norm(params):
    grad_norm = 0.0
    for p in params:
        if p.grad is not None and not torch.isnan(p.grad).any():
            grad_norm += torch.sum(p.grad**2)
        if torch.isnan(p.grad).any():
            grad_norm += torch.sum(torch.randn_like(p.grad)**2)
    return torch.sqrt(grad_norm)


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle=False, drop_last=False):
        # (T, B, ...) -> (T * B, ...)
        self.obs = {k: v.view(-1, *v.shape[2:]) for k, v in obs.items()}
        self.target_values = target_values.view(-1)
        self.N = self.target_values.shape[0]
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.N // self.batch_size
        else:
            self.length = ((self.N - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.N)
        self.obs = {k: v[index, ...] for k, v in self.obs.items()}
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.N)

        obs = {k: v[start_idx:end_idx, ...] for k, v in self.obs.items()}
        target_values = self.target_values[start_idx:end_idx]
        return obs, target_values


class SequentialCriticDataset:
    """
    Dataset for critic training that preserves temporal order for GRU-based encoders.
    Instead of random batching, it provides sequences in temporal order to maintain
    GRU hidden state consistency.
    """
    def __init__(self, obs, target_values, done_mask=None):
        # Keep original temporal structure (T, B, ...)
        self.obs = obs  # (T, B, ...)
        self.target_values = target_values  # (T, B)
        self.done_mask = done_mask  # (T, B) - indicates episode boundaries
        
        self.T, self.B = target_values.shape
        
    def __len__(self):
        return self.T  # Number of time steps
        
    def __getitem__(self, t):
        # Return data for time step t across all environments
        obs_t = {k: v[t] for k, v in self.obs.items()}  # (B, ...)
        target_values_t = self.target_values[t]  # (B,)
        
        if self.done_mask is not None:
            done_t = self.done_mask[t]  # (B,) - which envs finished episodes at step t
            return obs_t, target_values_t, done_t
        else:
            return obs_t, target_values_t
