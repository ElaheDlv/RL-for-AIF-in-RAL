# rl_baselines/cnn_extractors.py
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# cnn_extractors.py
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models

class SmallCNNSB3(BaseFeaturesExtractor):
    """
    Small CNN feature extractor for CARLA RL baselines.
    Matches the BC-like capacity for fair comparison.
    """

    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 512):
        super().__init__(observation_space, features_dim=out_dim)
        if len(observation_space.shape) != 3:
            raise ValueError("SmallCNNSB3 expects 3D image observations")
        C, H, W = observation_space.shape
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.net(th.zeros(1, C, H, W)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        # SB3 hands channel-first observations (C,H,W) after ensuring channel order.
        return self.fc(self.net(x))


class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
    ResNet18 pretrained on ImageNet, for RGB observations.
    Outputs a feature vector of given size.
    """
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 512, freeze: bool = True):
        super().__init__(observation_space, features_dim=out_dim)
        C, H, W = observation_space.shape
        assert C == 3, "ResNetFeatureExtractor expects RGB input (3 channels)"
        
        # Load pretrained ResNet
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # drop final fc
        self.backbone = nn.Sequential(*modules)  # feature extractor

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Final projection layer
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # ResNet expects (B,3,H,W) with float in [0,1] or normalized
        x = self.backbone(x)  # (B,512,1,1)
        x = th.flatten(x, 1)
        x = self.fc(x)
        return F.relu(x)


class GrayroadSmallCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN for grayscale road masks.
    """
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 256):
        super().__init__(observation_space, features_dim=out_dim)
        C, H, W = observation_space.shape
        assert C == 1, "GrayroadSmallCNN expects single-channel input"
        
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.net(th.zeros(1, C, H, W)).shape[1]
        self.fc = nn.Linear(n_flat, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return F.relu(self.fc(self.net(x)))
