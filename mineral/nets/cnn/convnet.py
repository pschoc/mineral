import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_size, channels, kernel_sizes, strides, norm_type='BatchNorm', act_type='ReLU'):
        super().__init__()
        h, w, in_channels = in_size
        
        self.act_fn = getattr(nn, act_type)()
        
        layers = []
        current_channels = in_channels
        current_h, current_w = h, w
        
        for ch, k, s in zip(channels, kernel_sizes, strides):
            # Conv layer
            layers.append(nn.Conv2d(current_channels, ch, k, stride=s, padding=k//2))
            
            # Normalization
            if norm_type == 'BatchNorm':
                layers.append(nn.BatchNorm2d(ch))
            elif norm_type == 'LayerNorm':
                # Calculate spatial dims after conv
                current_h = (current_h + k//2 * 2 - k) // s + 1
                current_w = (current_w + k//2 * 2 - k) // s + 1
                layers.append(nn.LayerNorm([ch, current_h, current_w]))
            
            # Activation
            layers.append(self.act_fn)
            current_channels = ch
        
        self.backbone = nn.Sequential(*layers)
        
        # Calculate final output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, h, w)
            dummy_out = self.backbone(dummy_input)
            self.out_dim = dummy_out.numel()
    
    def forward(self, x):
        # Input: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)
        return x.flatten(1)
