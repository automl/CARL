import copy
import math
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):

        weight_epsilon = self.weight_epsilon.to(x.device)
        bias_epsilon = self.bias_epsilon.to(x.device)

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class EvolvableCnnDQN(nn.Module):

    def __init__(self, input_shape: List[int],
                 channel_size: List[int],
                 kernal_size: List[int],
                 stride_size: List[int],
                 hidden_size: List[int],
                 num_actions: int,
                 num_atoms: int,
                 Vmin: int,
                 Vmax: int,
                 mlp_activation='relu',
                 cnn_activation='relu',
                 layer_norm=False, stored_values=None, device="cpu"):

        super(EvolvableCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.channel_size = channel_size
        self.kernal_size = kernal_size
        self.stride_size = stride_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.mlp_activation = mlp_activation
        self.cnn_activation = cnn_activation
        self.layer_norm = layer_norm
        self.device = device

        self.net = self.create_nets()
        self.feature_net, self.value_net, self.advantage_net = self.create_nets()

        if stored_values is not None:
            self.inject_parameters(pvec=stored_values, without_layer_norm=False)

    def get_activation(self, activation_names):

        activation_functions = {'tanh': nn.Tanh, 'gelu': nn.GELU, 'relu': nn.ReLU, 'elu': nn.ELU,
                                'softsign': nn.Softsign, 'sigmoid': nn.Sigmoid, 'softplus': nn.Softplus,
                                'lrelu': nn.LeakyReLU, 'prelu': nn.PReLU, }
        return activation_functions[activation_names]()

    def create_mlp(self, input_size, output_size, hidden_size, name):

        net_dict = OrderedDict()

        net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        if self.layer_norm:
            net_dict[f"{name}_layer_norm_0"] = nn.LayerNorm(hidden_size[0])
        net_dict[f"{name}_activation_0"] = self.get_activation(self.mlp_activation)

        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(hidden_size[l_no - 1], hidden_size[l_no])
                if self.layer_norm:
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.LayerNorm(hidden_size[l_no])
                net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(self.mlp_activation)
        net_dict[f"{name}_linear_layer_output"] = NoisyLinear(hidden_size[-1], output_size)
        return nn.Sequential(net_dict)

    def create_cnn(self, input_size, channel_size, kernal_size, stride_size, name):

        net_dict = OrderedDict()

        net_dict[f"{name}_conv_layer_0"] = nn.Conv2d(in_channels=input_size, out_channels=channel_size[0],
                                                     kernel_size=kernal_size[0],
                                                     stride=stride_size[0])
        if self.layer_norm:
            net_dict[f"{name}_layer_norm_0"] = nn.BatchNorm2d(channel_size[0])
        net_dict[f"{name}_activation_0"] = self.get_activation(self.cnn_activation)

        if len(channel_size) > 1:
            for l_no in range(1, len(channel_size)):
                net_dict[f"{name}_conv_layer_{str(l_no)}"] = nn.Conv2d(in_channels=channel_size[l_no - 1],
                                                                       out_channels=channel_size[l_no],
                                                                       kernel_size=kernal_size[l_no],
                                                                       stride=stride_size[l_no])
                if self.layer_norm:
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.BatchNorm2d(channel_size[l_no])
                net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(self.cnn_activation)

        return nn.Sequential(net_dict)

    def create_nets(self):

        feature_net = self.create_cnn(self.input_shape[0], self.channel_size, self.kernal_size, self.stride_size,
                                      name="feature")

        input_size = feature_net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

        value_net = self.create_mlp(input_size, output_size=self.num_atoms, hidden_size=self.hidden_size, name="value")
        advantage_net = self.create_mlp(input_size, output_size=self.num_atoms * self.num_actions,
                                        hidden_size=self.hidden_size,
                                        name="adcantage")

        feature_net.to(self.device)
        value_net.to(self.device)
        advantage_net.to(self.device)

        return feature_net, value_net, advantage_net

    def reset_noise(self):
        for l in self.value_net:
            if isinstance(l, NoisyLinear):
                l.reset_noise()
        for l in self.advantage_net:
            if isinstance(l, NoisyLinear):
                l.reset_noise()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        batch_size = x.size(0)
        x = x / 255.

        x = self.feature_net(x)
        x = x.view(batch_size, -1)

        value = self.value_net(x)
        advantage = self.advantage_net(x)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(-1, self.num_actions, self.num_atoms)

        return x

    def act(self, state):

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)

        state = state.to(self.device)

        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

    @property
    def short_dict(self):
        short_dict = {"channel_size": self.channel_size, "kernal_size": self.kernal_size,
                      "stride_size": self.stride_size, "hidden_size": self.hidden_size,
                      "num_atoms": self.num_atoms,
                      "Vmin": self.Vmin, "Vmax": self.Vmax,
                      "mlp_activation": self.mlp_activation, "cnn_activation": self.cnn_activation,
                      "layer_norm": self.layer_norm}
        return short_dict

    @property
    def init_dict(self):
        initdict = {"input_shape": self.input_shape, "channel_size": self.channel_size, "kernal_size": self.kernal_size,
                    "stride_size": self.stride_size, "hidden_size": self.hidden_size,
                    "num_actions": self.num_actions, "num_atoms": self.num_atoms,
                    "Vmin": self.Vmin, "Vmax": self.Vmax,
                    "mlp_activation": self.mlp_activation, "cnn_activation": self.cnn_activation,
                    "layer_norm": self.layer_norm, "device": self.device}
        return initdict

    def get_model_dict(self):

        model_dict = self.init_dict
        model_dict.update({'stored_values': self.extract_parameters(without_layer_norm=False)})
        return model_dict

    def count_parameters(self, without_layer_norm=False):
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or not 'layer_norm' in name:
                count += param.data.cpu().numpy().flatten().shape[0]
        return count

    def extract_grad(self, without_layer_norm=False):
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or not 'layer_norm' in name:
                sz = param.grad.data.cpu().numpy().flatten().shape[0]
                pvec[count:count + sz] = param.grad.data.cpu().numpy().flatten()
                count += sz
        return pvec.copy()

    def extract_parameters(self, without_layer_norm=False):
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or not 'layer_norm' in name:
                sz = param.data.cpu().detach().numpy().flatten().shape[0]
                pvec[count:count + sz] = param.data.cpu().detach().numpy().flatten()
                count += sz
        return copy.deepcopy(pvec)

    def inject_parameters(self, pvec, without_layer_norm=False):
        count = 0

        for name, param in self.named_parameters():
            if not without_layer_norm or not 'layer_norm' in name:
                sz = param.data.cpu().numpy().flatten().shape[0]
                raw = pvec[count:count + sz]
                reshaped = raw.reshape(param.data.cpu().numpy().shape)
                param.data = torch.from_numpy(copy.deepcopy(reshaped)).type(torch.FloatTensor)
                count += sz
        return pvec

    def add_mlp_layer(self):
        if len(self.hidden_size) < 3:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]

            self.recreate_nets()
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]

        if self.hidden_size[hidden_layer] + numb_new_nodes <= 1024:  # HARD LIMIT

            self.hidden_size[hidden_layer] += numb_new_nodes

            self.recreate_nets()
        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def add_cnn_layer(self):
        if len(self.channel_size) < 6:  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            self.kernal_size += [3]

            stride_size_list = [[4], [4, 2], [4, 2, 1], [2, 2, 2, 1], [2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]]
            self.stride_size = stride_size_list[len(self.channel_size) - 1]

            self.recreate_nets()
        else:
            self.add_cnn_channel()

    def change_cnn_kernal(self):
        if len(self.channel_size) > 1:
            hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[0]
            self.kernal_size[hidden_layer] = np.random.choice([3, 4, 5, 7])

            self.recreate_nets()
        else:
            self.add_cnn_layer()

    def add_cnn_channel(self, hidden_layer=None, numb_new_channels=None):

        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)
        if numb_new_channels is None:
            numb_new_nodes = np.random.choice([8, 16, 32], 1)[0]

        if self.channel_size[hidden_layer] + numb_new_nodes <= 256:  # HARD LIMIT

            self.channel_size[hidden_layer] += numb_new_nodes

            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_nets(self):
        new_feature_net, new_value_net, new_advantage_net = self.create_nets()
        new_feature_net = self.preserve_parameters(old_net=self.feature_net, new_net=new_feature_net)
        new_value_net = self.preserve_parameters(old_net=self.value_net, new_net=new_value_net)
        new_advantage_net = self.preserve_parameters(old_net=self.advantage_net, new_net=new_advantage_net)
        self.feature_net, self.value_net, self.advantage_net = new_feature_net, new_value_net, new_advantage_net

    def clone(self):
        clone = EvolvableCnnDQN(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
        return clone

    def preserve_parameters(self, old_net, new_net):

        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if not "norm" in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[:min(old_size[0], new_size[0])] = old_net_dict[key].data[
                                                                         :min(old_size[0], new_size[0])]
                        elif len(param.data.size()) == 2:
                            param.data[:min(old_size[0], new_size[0]), :min(old_size[1], new_size[1])] = old_net_dict[
                                                                                                             key].data[
                                                                                                         :min(old_size[
                                                                                                                  0],
                                                                                                              new_size[
                                                                                                                  0]),
                                                                                                         :min(old_size[
                                                                                                                  1],
                                                                                                              new_size[
                                                                                                                  1])]
                        else:
                            param.data[:min(old_size[0], new_size[0]), :min(old_size[1], new_size[1]),
                            :min(old_size[2], new_size[2]),
                            :min(old_size[3], new_size[3])] = old_net_dict[key].data[
                                                              :min(old_size[0], new_size[0]),
                                                              :min(old_size[1], new_size[1]),
                                                              :min(old_size[2], new_size[2]),
                                                              :min(old_size[3], new_size[3]),
                                                              ]

        return new_net

    def shrink_preserve_parameters(self, old_net, new_net):

        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if not "norm" in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[:min_0, :min_1]
        return new_net
