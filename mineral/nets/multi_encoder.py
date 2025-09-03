import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cnn as CNN
from . import pcd as PCD
from .mlp import MLP


class MultiEncoder(nn.Module):
    def __init__(self, obs_space, cfg, weight_init_fn=None):
        super().__init__()
        cnn_keys = cfg.get('cnn_keys', '$^')
        pcd_keys = cfg.get('pcd_keys', '$^')
        mlp_keys = cfg.get('mlp_keys', '^obs$')
        concat_keys = cfg.get('concat_keys', '^cnn$|^pcd$|^mlp$')
        
        # GRU parameters
        self.use_gru = cfg.get('use_gru', False)
        self.gru_hidden_size = cfg.get('gru_hidden_size', None)
        self.gru_num_layers = cfg.get('gru_num_layers', 1)
        self.gru_bidirectional = cfg.get('gru_bidirectional', False)
        self.gru_batch_first = cfg.get('gru_batch_first', True)
        self.gru_dropout = cfg.get('gru_dropout', 0.0)

        excluded = {}
        shapes = {k: v for k, v in obs_space.items() if k not in excluded and not k.startswith('info_')}
        print('Encoder input shapes:', shapes)
        self.cnn_shapes = {k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))}
        self.pcd_shapes = {k: v for k, v in shapes.items() if (len(v) == 2 and re.match(pcd_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.pcd_shapes, **self.mlp_shapes}
        self.concat_keys = re.compile(concat_keys)
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder PCD shapes:', self.pcd_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)

        self.out_dim = 0
        if self.cnn_shapes:
            in_channels = sum([v[-1] for v in self.cnn_shapes.values()])
            some_shape = list(self.cnn_shapes.values())[0]
            in_size = (some_shape[0], some_shape[1], in_channels)

            cnn, cnn_kwargs = cfg.cnn, cfg.cnn_kwargs
            Cls = getattr(CNN, cnn)
            self._cnn = Cls(in_size, **cnn_kwargs)
            self.out_dim += self._cnn.out_dim

        if self.pcd_shapes:
            pcd_inputs_kwargs = cfg.get('pcd_inputs_kwargs', {})
            self._pcd_inputs = PCD.PCDInputs(self.pcd_shapes, **pcd_inputs_kwargs)

            pcd, pcd_kwargs = cfg.pcd, cfg.pcd_kwargs
            Cls = getattr(PCD, pcd)
            self._pcd = Cls(self.pcd_shapes, **pcd_kwargs)
            self.out_dim += self._pcd.global_feature_dim

        if self.mlp_shapes:
            tensor_shape = sum([np.prod(v, dtype=int) for v in self.mlp_shapes.values()])

            mlp_kwargs = cfg.get('mlp_kwargs', None)
            if mlp_kwargs is not None:
                self._mlp = MLP(tensor_shape, **mlp_kwargs)
                self.out_dim += self._mlp.out_dim
            else:
                self._mlp = None
                self.out_dim += tensor_shape

        print('Encoder out_dim:', self.out_dim)
        
        # Initialize GRU if requested
        self.gru_hidden_states = None
        if self.use_gru:
            if self.gru_hidden_size is None:
                self.gru_hidden_size = self.out_dim  # Default to combined embedding dimension
            
            # GRU dropout only applies if num_layers > 1
            gru_dropout = self.gru_dropout if self.gru_dropout > 0.0 and self.gru_num_layers > 1 else 0.0
            
            self.gru = nn.GRU(
                input_size=self.out_dim,
                hidden_size=self.gru_hidden_size,
                num_layers=self.gru_num_layers,
                bias=True,
                batch_first=self.gru_batch_first,
                dropout=gru_dropout,
                bidirectional=self.gru_bidirectional
            )
            
            # Update output dimension to GRU output size
            self.out_dim = self.gru_hidden_size * (2 if self.gru_bidirectional else 1)
        else:
            self.gru = None
        
        print('Final Encoder out_dim (after GRU):', self.out_dim)

        self.weight_init = cfg.get('weight_init', None)
        self.weight_init_cnn = cfg.get('weight_init_cnn', None)
        self.weight_init_pcd = cfg.get('weight_init_pcd', None)
        self.weight_init_mlp = cfg.get('weight_init_mlp', None)
        self.weight_init_fn = weight_init_fn
        self.reset_parameters()

    def reset_parameters(self):
        if self.cnn_shapes:
            if hasattr(self._cnn, "reset_parameters"):
                self._cnn.reset_parameters()
        if self.pcd_shapes:
            if hasattr(self._pcd, "reset_parameters"):
                self._pcd.reset_parameters()
        if self.mlp_shapes:
            if self._mlp is not None:
                if hasattr(self._mlp, "reset_parameters"):
                    self._mlp.reset_parameters()

        if self.weight_init_fn is not None:
            if self.weight_init is not None:
                self.weight_init_fn(self, self.weight_init)

            if self.weight_init_cnn is not None and self.cnn_shapes:
                self.weight_init_fn(self._cnn, self.weight_init_cnn)

            if self.weight_init_pcd is not None and self.pcd_shapes:
                self.weight_init_fn(self._pcd, self.weight_init_pcd)

            if self.weight_init_mlp is not None and self.mlp_shapes:
                self.weight_init_fn(self._mlp, self.weight_init_mlp)
                
        # Initialize GRU parameters if present
        if self.use_gru and hasattr(self, 'gru'):
            if self.weight_init_fn is not None and self.weight_init is not None:
                self.weight_init_fn(self.gru, self.weight_init)

    def init_gru_states(self, batch_size, device=None):
        """Initialize GRU hidden states for all environments."""
        if not self.use_gru or self.gru is None:
            return None
            
        num_directions = 2 if self.gru_bidirectional else 1
        hidden_shape = (self.gru_num_layers * num_directions, batch_size, self.gru_hidden_size)
        
        if device is None:
            return torch.zeros(hidden_shape)
        else:
            return torch.zeros(hidden_shape, device=device)
            
    def reset_gru_states(self, env_ids=None, device=None):
        """Reset GRU hidden states for specific environments or all environments."""
        if not self.use_gru or self.gru_hidden_states is None:
            return
            
        if device is None:
            device = self.gru_hidden_states.device
            
        if env_ids is None:
            # Reset all environments
            batch_size = self.gru_hidden_states.shape[1]  # (num_layers*num_directions, batch_size, hidden_size)
            self.gru_hidden_states = self.init_gru_states(batch_size, device)
        else:
            # Reset specific environments
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().numpy()
            elif not isinstance(env_ids, (list, tuple, np.ndarray)):
                env_ids = [env_ids]
                
            # Zero out hidden states for specified environments
            self.gru_hidden_states[:, env_ids, :] = 0.0

    def cnn(self, x):
        inputs = torch.cat([x[k] for k in self.cnn_shapes], -1)
        output = self._cnn(inputs)
        output = output.reshape((output.shape[0], -1))
        return output

    def pcd(self, x):
        inputs = {k: x[k] for k in self.pcd_shapes}
        data = self._pcd_inputs(inputs)
        outputs = self._pcd(data)
        global_z, local_z = outputs
        return global_z, local_z

    def mlp(self, x):
        inputs = [x[k][..., None] if len(self.shapes[k]) == 0 else x[k] for k in self.mlp_shapes]
        inputs = torch.cat([x.reshape(x.shape[0], -1) for x in inputs], dim=-1)
        if self._mlp is not None:
            output = self._mlp(inputs)
        else:
            output = inputs
        return output

    def forward(self, x, return_gru_output=False):
        """
        Forward pass through encoder.
        
        Args:
            x: Input observations dict
            return_gru_output: If True and use_gru=True, return (outputs, gru_hidden_state) tuple
            
        Returns:
            outputs: Dict with processed features
            If return_gru_output=True and use_gru=True: (outputs, gru_hidden_state) tuple
        """
        outputs = {}
        if self.cnn_shapes:
            outputs['cnn'] = self.cnn(x)

        if self.pcd_shapes:
            outputs['pcd'], outputs['pcd_local'] = self.pcd(x)

        if self.mlp_shapes:
            outputs['mlp'] = self.mlp(x)

        # Concatenate features before GRU
        z = torch.cat([v for k, v in outputs.items() if re.match(self.concat_keys, k)], dim=-1)
        
        if self.use_gru and self.gru is not None:
            # Initialize GRU states if not already done
            if self.gru_hidden_states is None:
                batch_size = z.shape[0]
                device = z.device
                self.gru_hidden_states = self.init_gru_states(batch_size, device)
            
            # If input is 2D (batch_size, features), expand to (batch_size, 1, features) for GRU
            if len(z.shape) == 2:
                z = z.unsqueeze(1)  # Add sequence dimension
            
            # Forward pass through GRU
            gru_output, self.gru_hidden_states = self.gru(z, self.gru_hidden_states)
            
            if self.gru_batch_first:
                # Take the last time step: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
                z = gru_output[:, -1, :]
            else:
                # Take the last time step: (seq_len, batch_size, hidden_size) -> (batch_size, hidden_size)
                z = gru_output[-1, :, :]
                
        if torch.isnan(z).any():
            nan_mask = torch.isnan(z)
            z = torch.where(nan_mask, torch.randn_like(z), z)
        
        outputs['z'] = z
        
        if return_gru_output and self.use_gru:
            return outputs, self.gru_hidden_states
        return outputs
