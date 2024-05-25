import torch
import torch.nn as nn
from utils.setSeed import setSeed

# Set seed for reproducibility
setSeed(42)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class bigff(nn.Module):
    '''Simplified Bi-directional Gated Feature Fusion using Depthwise Separable Convolutions.'''

    def __init__(self, in_channels, out_channels):
        super(bigff, self).__init__()

        self.structure_gate = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels * 2, out_channels=out_channels),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels * 2, out_channels=out_channels),
            nn.Sigmoid()
        )

    def forward(self, texture_feature, structure_feature):
        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + gate_structure_to_texture * structure_feature
        structure_feature = structure_feature + gate_texture_to_structure * texture_feature

        return torch.cat((texture_feature, structure_feature), dim=1)