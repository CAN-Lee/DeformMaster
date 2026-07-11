import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        # [STABILITY] Use GroupNorm instead of BatchNorm to support batch_size=1
        self.gn1 = nn.GroupNorm(8, 64)
        self.gn2 = nn.GroupNorm(8, 128)
        self.gn3 = nn.GroupNorm(32, 1024)
        self.gn4 = nn.GroupNorm(32, 512)
        self.gn5 = nn.GroupNorm(16, 256)

    def forward(self, x, x_mask=None):
        batchsize = x.size()[0]
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(1) # [B, 1, N]
        
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))

        if x_mask is not None:
            # Masked max pooling
            x = x * x_mask - (~x_mask) * 1e10
            x = torch.max(x, 2, keepdim=True)[0]
        else:
            x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.gn4(self.fc1(x)))
        x = F.relu(self.gn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_dim=64, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        # [STABILITY] Use GroupNorm
        self.gn1 = nn.GroupNorm(8, 64)
        self.gn2 = nn.GroupNorm(8, 128)
        self.gn3 = nn.GroupNorm(8, feature_dim)
        
        self.feature_dim = feature_dim
        self.global_feat = global_feat

    def forward(self, x, x_mask=None):
        B, D, N = x.size()
        if x_mask is None:
            x_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        # 1. Spatial Transformation
        trans = self.stn(x, x_mask)
        x_pts = x.transpose(2, 1) # [B, N, D]
        
        # Transform only the first 3 dims (pos)
        pos = x_pts[:, :, :3]
        if D > 3:
            others = x_pts[:, :, 3:]
        
        pos = torch.bmm(pos, trans)
        
        if D > 3:
            x = torch.cat([pos, others], dim=2)
        else:
            x = pos
        x = x.transpose(2, 1) # [B, D, N]

        # 2. Feature Extraction
        x = F.relu(self.gn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))

        if not self.global_feat:
            # Return local features per particle [B, feature_dim, N]
            return x, trans
        
        # 3. Global Feature (Max Pooling)
        x = x * x_mask.unsqueeze(1) - (~x_mask.unsqueeze(1)) * 1e10
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feature_dim) # [B, feature_dim]
        
        return x, trans
