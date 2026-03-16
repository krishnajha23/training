import torch
import torch.nn as nn
import torch.nn.functional as F

class TowerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, embed_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim,
                 hidden_dims=(256, 128), embed_dim=64):
        super().__init__()
        self.user_tower  = TowerEncoder(user_dim,  hidden_dims, embed_dim)
        self.item_tower  = TowerEncoder(item_dim,  hidden_dims, embed_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, user_features, item_features):
        return (self.user_tower(user_features),
                self.item_tower(item_features))


def build_two_tower(user_dim=34, item_dim=34,
                    hidden_dims=(256, 128), embed_dim=64):
    model = TwoTowerModel(user_dim, item_dim, hidden_dims, embed_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Two-Tower parameters: {total_params/1e3:.1f}K")
    return model