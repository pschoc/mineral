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

        # GRU options
        self.use_gru = cfg.get('use_gru', False)
        self.gru_hidden_size = cfg.get('gru_hidden_size', None)
        self.gru_num_layers = cfg.get('gru_num_layers', 1)
        self.gru_bidirectional = cfg.get('gru_bidirectional', False)        
        self.gru_bias = cfg.get('gru_bias', True)
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

        out_dim = 0
        if self.cnn_shapes:
            in_channels = sum([v[-1] for v in self.cnn_shapes.values()])
            some_shape = list(self.cnn_shapes.values())[0]
            in_size = (some_shape[0], some_shape[1], in_channels)

            cnn, cnn_kwargs = cfg.cnn, cfg.cnn_kwargs
            Cls = getattr(CNN, cnn)
            self._cnn = Cls(in_size, **cnn_kwargs)
            out_dim += self._cnn.out_dim

        if self.pcd_shapes:
            pcd_inputs_kwargs = cfg.get('pcd_inputs_kwargs', {})
            self._pcd_inputs = PCD.PCDInputs(self.pcd_shapes, **pcd_inputs_kwargs)

            pcd, pcd_kwargs = cfg.pcd, cfg.pcd_kwargs
            Cls = getattr(PCD, pcd)
            self._pcd = Cls(self.pcd_shapes, **pcd_kwargs)
            out_dim += self._pcd.global_feature_dim

        if self.mlp_shapes:
            tensor_shape = sum([np.prod(v, dtype=int) for v in self.mlp_shapes.values()])

            mlp_kwargs = cfg.get('mlp_kwargs', None)
            if mlp_kwargs is not None:
                self._mlp = MLP(tensor_shape, **mlp_kwargs)
                out_dim += self._mlp.out_dim
            else:
                self._mlp = None
                out_dim += tensor_shape

        # Initialize GRU if requested
        if self.use_gru:
            self._gru = nn.GRU(
                input_size=out_dim,
                hidden_size=self.gru_hidden_size,
                num_layers=self.gru_num_layers,
                bias=self.gru_bias,
                batch_first=True,
                dropout=self.gru_dropout if self.gru_num_layers > 1 else 0.0,
                bidirectional=self.gru_bidirectional
            )
            
            # Initialize hidden state
            self.gru_hidden = None
            
            # GRU output dimension
            gru_output_dim = self.gru_hidden_size * (2 if self.gru_bidirectional else 1)
            
            # Add MLP to reduce GRU output to desired size
            gru_out_size = cfg.get('gru_out_size', gru_output_dim)
            if gru_out_size != gru_output_dim:
                self._gru_mlp = nn.Linear(gru_output_dim, gru_out_size)
                self.out_dim = gru_out_size
            else:
                self._gru_mlp = None
                self.out_dim = gru_output_dim
        else:
            self._gru = None
            self._gru_mlp = None
            self.gru_hidden = None
            self.out_dim = out_dim

        print('Encoder out_dim:', self.out_dim)

        self.weight_init = cfg.get('weight_init', None)
        self.weight_init_cnn = cfg.get('weight_init_cnn', None)
        self.weight_init_pcd = cfg.get('weight_init_pcd', None)
        self.weight_init_mlp = cfg.get('weight_init_mlp', None)
        if self.use_gru:
            self.weight_init_gru = cfg.get('weight_init_gru', None)
        self.weight_init_fn = weight_init_fn

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init_fn is not None:
            if self.weight_init is not None:
                self.weight_init_fn(self, self.weight_init)

            if self.weight_init_cnn is not None and self.cnn_shapes:
                self.weight_init_fn(self._cnn, self.weight_init_cnn)

            if self.weight_init_pcd is not None and self.pcd_shapes:
                self.weight_init_fn(self._pcd, self.weight_init_pcd)

            if self.weight_init_mlp is not None and self.mlp_shapes:
                self.weight_init_fn(self._mlp, self.weight_init_mlp)

            if self.use_gru and self.weight_init_gru is not None:
                self.weight_init_fn(self._gru, self.weight_init_gru)

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
    
    def gru(self, x):
        """Process input through GRU if enabled, otherwise return input unchanged."""
        if self.use_gru and self._gru is not None:
            # Ensure input has sequence dimension (batch_size, seq_len, features)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            # Pass through GRU
            try:
                gru_out, self.gru_hidden = self._gru(x, self.gru_hidden)
            except Exception as e:
                print("Error occurred in GRU:", e)
                gru_out = x  # Fallback to original input on error

            # Take the last output from the sequence
            gru_out = gru_out[:, -1, :]  # (batch_size, hidden_size * num_directions)
            
            # Apply optional MLP projection
            if self._gru_mlp is not None:
                gru_out = self._gru_mlp(gru_out)
                
            return gru_out
        else:
            return x
    
    def reset_gru_hidden(self, batch_size_or_done_ids=None, device=None):
        """Reset the GRU hidden state.
        
        Args:
            batch_size_or_done_ids: Can be either:
                - int: batch size to reset all hidden states
                - torch.Tensor: tensor of environment IDs to reset selectively
                - None: reset to None
            device: device to create tensors on
        """
        if self.use_gru and self._gru is not None:
            if batch_size_or_done_ids is None:
                self.gru_hidden = None
            elif isinstance(batch_size_or_done_ids, int):
                # Reset all hidden states with given batch size
                batch_size = batch_size_or_done_ids
                if device is not None:
                    num_directions = 2 if self.gru_bidirectional else 1
                    self.gru_hidden = torch.zeros(
                        self.gru_num_layers * num_directions,
                        batch_size,
                        self.gru_hidden_size,
                        device=device
                    )
                else:
                    self.gru_hidden = None
            else:
                # Reset specific environment IDs
                done_env_ids = batch_size_or_done_ids
                if self.gru_hidden is not None and len(done_env_ids) > 0:
                    # Reset only the hidden states for done environments
                    self.gru_hidden[:, done_env_ids, :] = 0.0
                elif self.gru_hidden is None:
                    # If hidden state doesn't exist yet, we can't selectively reset
                    print("Warning: Cannot selectively reset GRU hidden states - hidden state is None")
    def forward(self, x):
        outputs = {}
        if self.cnn_shapes:
            outputs['cnn'] = self.cnn(x)

        if self.pcd_shapes:
            outputs['pcd'], outputs['pcd_local'] = self.pcd(x)

        if self.mlp_shapes:
            outputs['mlp'] = self.mlp(x)

        concat_in = torch.cat([v for k, v in outputs.items() if re.match(self.concat_keys, k)], dim=-1)

        if self.use_gru:
            z = self.gru(concat_in)
        else:
            z = concat_in
        
        outputs['z'] = z
        return outputs