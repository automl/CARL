# From https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common import layer_init
from torch.distributions.categorical import Categorical
from gym import spaces

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class PPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # TODO(frederik): handle dict observation space with context
        if isinstance(envs.single_observation_space, spaces.Dict):
            shape = envs.single_observation_space["state"].shape
            self.context_shape = envs.single_observation_space["context"].shape
        else:
            shape = envs.single_observation_space.shape
            self.context_shape = None
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        if self.context_shape is not None:
            shape = self.context_shape
            context_conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                context_conv_seqs.append(conv_seq)
            self.context_embedding = nn.Sequential(
                *context_conv_seqs,
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
            )
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.get_action_and_value(x)

    def get_value(self, x):
        if isinstance(x, dict):
            state = x["state"]
            context = x["context"].float()
        else:
            state = x
            context = None
        hidden = self.network(state / 255.0)
        if self.context_shape is not None:
            hidden = hidden + self.context_embedding(context)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, dict):
            state = x["state"]
            context = x["context"].float()
        else:
            state = x
            context = None
        hidden = self.network(state / 255.0)
        if self.context_shape is not None:
            hidden = hidden + self.context_embedding(context)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
