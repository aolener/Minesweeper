"""Fully convolutional Q-network: (B, 11, H, W) -> (B, H*W)."""
import torch
import torch.nn as nn
 
 
class MinesweeperDQN(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        num_layers: int = 6,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(num_layers):
            # 3x3 with padding=1 preserves H and W.
            layers += [nn.Conv2d(c, hidden_channels, 3, padding=1), nn.ReLU(inplace=True)]
            c = hidden_channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)  # one Q-value per tile
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,1,H,W) -> (B,H*W); flat index = r*W + c, matching the env's action id.
        return self.head(self.body(x)).flatten(1)
 
 
def mask_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Q-hat: keep Q where mask is True, -inf elsewhere."""
    return q.masked_fill(~mask, float("-inf"))
 
 
def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask_q(q, mask).argmax(dim=1)