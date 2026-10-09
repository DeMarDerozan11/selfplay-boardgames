"""Small residual conv net with a policy head and a value head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch)
        self.c2, self.b2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class PolicyValueNet(nn.Module):
    def __init__(self, in_ch, height, width, num_actions, channels=64, blocks=3):
        super().__init__()
        self.config = dict(in_ch=in_ch, height=height, width=width, num_actions=num_actions,
                           channels=channels, blocks=blocks)
        self.stem = nn.Sequential(nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * height * width, num_actions)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.Linear(height * width, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(self.stem(x))
        logits = self.p_fc(self.p_conv(h).flatten(1))
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(self.v_conv(h).flatten(1))))).squeeze(1)
        return logits, v

    @classmethod
    def from_checkpoint(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(**ck["config"])
        net.load_state_dict(ck["model"])
        return net.eval()
