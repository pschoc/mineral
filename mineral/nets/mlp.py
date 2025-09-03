import torch
import torch.nn as nn


def Norm(norm_type, size, **norm_kwargs):
    if norm_type is None:
        return nn.Identity()

    module = torch.nn
    Cls = getattr(module, norm_type)
    norm = Cls(size, **norm_kwargs)
    return norm


def Act(act_type, **act_kwargs):
    if act_type is None:
        return nn.Identity()

    module = torch.nn.modules.activation
    Cls = getattr(module, act_type)
    act = Cls(**act_kwargs)
    return act


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        units=[512, 256, 128],
        dropout=None,
        dropout_kwargs=dict(inplace=False),
        where_dropout="every",
        norm_type=None,
        norm_kwargs={},
        act_type="ReLU",
        act_kwargs=dict(inplace=True),
        bias=True,
        plain_last=False,
        # GRU options
        use_gru=False,
        gru_hidden_size=None,
        gru_num_layers=1,
        gru_bidirectional=False,
        gru_batch_first=True,
    ):
        super().__init__()
        if out_dim is not None:
            units = [*units, out_dim]
        self.in_dim = in_dim
        self.out_dim = units[-1]
        self.units = units
        self.use_gru = use_gru
        self.gru_batch_first = gru_batch_first

        # Initialize GRU if requested
        if use_gru:
            if gru_hidden_size is None:
                gru_hidden_size = in_dim  # Default to input dimension
            
            self.gru_hidden_size = gru_hidden_size
            self.gru_num_layers = gru_num_layers
            self.gru_bidirectional = gru_bidirectional
            
            # GRU dropout only applies if num_layers > 1
            gru_dropout = dropout if dropout is not None and gru_num_layers > 1 else 0.0
            
            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                bias=bias,
                batch_first=gru_batch_first,
                dropout=gru_dropout,
                bidirectional=gru_bidirectional
            )
            
            # Update input dimension for MLP to GRU output size
            mlp_input_dim = gru_hidden_size * (2 if gru_bidirectional else 1)
        else:
            self.gru = None
            mlp_input_dim = in_dim

        # Build MLP layers
        in_size = mlp_input_dim
        layers = []
        for i, out_size in enumerate(units):
            lin = nn.Linear(in_size, out_size, bias=bias)
            layers.append(lin)
            if plain_last and i == len(units) - 1:
                break

            if dropout is not None:
                add_dropout = False
                if i == 0 and where_dropout in ("every", "first"):
                    add_dropout = True
                if (i != 0 and i != len(units) - 1) and where_dropout in ("every"):
                    add_dropout = True
                if i == len(units) - 1 and where_dropout in ("every", "last"):
                    add_dropout = True
                if add_dropout:
                    dp = nn.Dropout(dropout, **dropout_kwargs)
                    layers.append(dp)
            if norm_type is not None:
                norm = Norm(norm_type, out_size, **norm_kwargs)
                layers.append(norm)
            if act_type is not None:
                act = Act(act_type, **act_kwargs)
                layers.append(act)
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        """
        Forward pass through optional GRU and MLP layers.
        
        Args:
            x: Input tensor. If use_gru=True, expects shape:
               - (batch_size, seq_len, in_dim) if gru_batch_first=True
               - (seq_len, batch_size, in_dim) if gru_batch_first=False
               If use_gru=False, expects (batch_size, in_dim)
            hidden: Initial hidden state for GRU (optional, only used if use_gru=True)
            
        Returns:
            If use_gru=True: (output, hidden_state)
            If use_gru=False: output
        """
        if self.use_gru and self.gru is not None:
            # Pass through GRU
            try:
                gru_output, hidden_state = self.gru(x, hidden)
            except Exception as e:
                print(f"Error occurred in GRU layer: {e}")
                return None

            if self.gru_batch_first:
                # Take the last time step: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
                last_output = gru_output[:, -1, :]
            else:
                # Take the last time step: (seq_len, batch_size, hidden_size) -> (batch_size, hidden_size)
                last_output = gru_output[-1, :, :]
            
            # Pass through MLP
            mlp_output = self.mlp(last_output)
            return mlp_output, hidden_state
        else:
            # Standard MLP forward pass
            return self.mlp(x)
    
    def init_hidden(self, batch_size, device=None):
        """Initialize hidden state for the GRU (only relevant if use_gru=True)."""
        if not self.use_gru or self.gru is None:
            return None
            
        num_directions = 2 if self.gru_bidirectional else 1
        hidden_shape = (self.gru_num_layers * num_directions, batch_size, self.gru_hidden_size)
        
        if device is None:
            return torch.zeros(hidden_shape)
        else:
            return torch.zeros(hidden_shape, device=device)
