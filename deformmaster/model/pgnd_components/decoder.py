import torch
import torch.nn as nn
import torch.nn.functional as F

class CondNeRFModel(nn.Module):
    """
    Neural Field Decoder that takes spatial location (Eulerian) 
    and conditional features (Lagrangian) to predict properties (delta_v).
    """
    def __init__(self, xyz_dim, condition_dim, out_channel=3, num_layers=4, hidden_size=128, skip_connect_every=4):
        super(CondNeRFModel, self).__init__()
        self.xyz_dim = xyz_dim
        self.condition_dim = condition_dim
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_connect_every = skip_connect_every
        
        # 1. Feature fusion (XYZ + Condition)
        self.fusion = nn.Linear(xyz_dim + condition_dim, hidden_size)
        
        # 2. Sequential MLP
        layers = []
        for i in range(num_layers - 1):
            if i % skip_connect_every == 0 and i != 0:
                layers.append(nn.Linear(hidden_size + (xyz_dim + condition_dim), hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        
        # 3. Final projection
        self.output_layer = nn.Linear(hidden_size, out_channel)

    def forward(self, xyz, condition):
        """
        Args:
            xyz: Positional encoding of spatial locations [B, N_pts, xyz_dim]
            condition: Conditional features [B, N_pts, condition_dim]
        Returns:
            Predict [B, N_pts, out_channel]
        """
        x_in = torch.cat([xyz, condition], dim=-1) # [B, N, D_in]
        
        x = F.relu(self.fusion(x_in))
        
        for i, layer in enumerate(self.layers):
            if i % self.skip_connect_every == 0 and i != 0:
                x = torch.cat([x, x_in], dim=-1)
            x = F.relu(layer(x))
            
        return self.output_layer(x)
