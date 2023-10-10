from typing import List, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None, units=[512, 256, 128], act_type="ELU", norm_type=None):
        super().__init__()
        units = units = [out_dim] if out_dim is not None else units
        layers = []
        for output_size in units:
            layers.append(nn.Linear(in_dim, output_size))
            if norm_type is not None:
                module = torch.nn
                Cls = getattr(module, norm_type)
                layers.append(Cls(output_size))
            if act_type is not None:
                module = torch.nn.modules.activation
                Cls = getattr(module, act_type)
                layers.append(Cls())
            in_dim = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def create_simple_mlp(in_dim, out_dim, hidden_layers, act_type="ELU", act_kwargs=dict(inplace=True)):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            module = torch.nn.modules.activation
            Cls = getattr(module, act_type)
            act = Cls(**act_kwargs)
            model.append(act)
    return nn.Sequential(*model)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim, out_dim=out_dim, hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class TanhMLPPolicy(MLPNet):
    def forward(self, state):
        return super().forward(state).tanh()


class DoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)  # two Q values

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


class EnsembleQ(nn.Module):
    def __init__(self, state_dim, act_dim, n_critics=2):
        super().__init__()
        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            critics.append(MLPNet(in_dim=state_dim + act_dim, out_dim=1))
        self.critics = nn.ModuleList(critics)

    def forward(self, state: Tensor, action: Tensor) -> List[Tensor]:
        input_x = torch.cat((state, action), dim=1)
        qvalue_list = [critic(input_x) for critic in self.critics]
        return qvalue_list


class DistributionalDoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, device="cuda"):
        super().__init__()
        self.device = device
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        Q1, Q2 = self.get_q1_q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)  # two Q values

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1)
