from typing import Dict, List, Tuple, Type, Union, Optional, Any

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs, get_flattened_obs_dim
from stable_baselines3.td3.policies import TD3Policy, Actor, ContinuousCritic, CombinedExtractor
from stable_baselines3.common.type_aliases import TensorDict, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


def get_state_seq(n_state_features: int, context_branch_width: int):
    state_seq = nn.Sequential(
        nn.Linear(n_state_features, context_branch_width),
        nn.ReLU()
    )
    return state_seq


def get_context_seq(n_context_features: int, context_branch_width: int):
    context_seq = nn.Sequential(
        nn.Linear(n_context_features, context_branch_width),
        nn.ReLU(),
        nn.Linear(context_branch_width, context_branch_width),
        nn.Sigmoid(),
    )
    return context_seq


def get_seqs(n_state_features: int, n_context_features: int, context_branch_width: int = 256):
    state_seq = get_state_seq(n_state_features=n_state_features, context_branch_width=context_branch_width)
    context_seq = get_context_seq(n_context_features=n_context_features, context_branch_width=context_branch_width)
    return state_seq, context_seq


class CGateFeatureExtractor(nn.Module):
    def __init__(self, n_state_features: int, n_context_features: int, context_branch_width: int = 256):
        super().__init__()
        state_seq, context_seq = get_seqs(
            n_state_features=n_state_features,
            n_context_features=n_context_features,
            context_branch_width=context_branch_width
        )

        extractors = {}
        extractors["context"] = context_seq
        extractors["state"] = state_seq
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.mul(*encoded_tensor_list)


def get_actor_head(action_dim: int, context_branch_width: int = 256, head_width: int = 256):
    actor_head = nn.Sequential(
        nn.Linear(context_branch_width, head_width),
        nn.ReLU(),
        nn.Linear(head_width, action_dim)  # TODO: no zero init?
    )
    return actor_head


def get_critic_head(context_branch_width: int = 256, head_width: int = 256):
    critic_head = nn.Sequential(
        nn.Linear(context_branch_width, head_width),
        nn.ReLU(),
        nn.Linear(head_width, 1)  # TODO: no zero init?
        # TODO why ravel? use here as well?
    )
    return critic_head


class CGateActor(Actor):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        context_branch_width: int = 256,
        head_width: int = 256,
        **kwargs
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        n_state_features = get_flattened_obs_dim(observation_space["state"])
        n_context_features = get_flattened_obs_dim(observation_space["context"])
        observation_extractor = CGateFeatureExtractor(
            n_state_features=n_state_features,
            n_context_features=n_context_features,
            context_branch_width=context_branch_width
        )
        actor_head = get_actor_head(
            action_dim=action_dim, context_branch_width=context_branch_width, head_width=head_width)

        self.mu = nn.Sequential(
            observation_extractor,
            actor_head
        )


class CGateCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        context_branch_width: int = 256,
        head_width: int = 256,
        **kwargs
    ):
        BaseModel.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        n_state_features = get_flattened_obs_dim(observation_space["state"])
        n_context_features = get_flattened_obs_dim(observation_space["context"])

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            observation_extractor = CGateFeatureExtractor(
                n_state_features=n_state_features + action_dim,
                n_context_features=n_context_features,
                context_branch_width=context_branch_width
            )
            critic_head = get_critic_head(
                context_branch_width=context_branch_width, head_width=head_width)
            q_net = nn.Sequential(
                observation_extractor,
                critic_head
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: TensorDict, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # copy observations because obs are a reference
        # if not copied, the actor will receive the concatenated vector of state and action
        qvalue_input = obs.copy()
        qvalue_input["state"] = th.cat([qvalue_input["state"], actions], dim=1)  # concat actions to state
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: TensorDict, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        # with th.no_grad():
        #     features = self.extract_features(obs)
        qvalue_input = obs.copy()
        qvalue_input["state"] = th.cat([qvalue_input["state"], actions], dim=1)  # concat actions to state
        return self.q_networks[0](qvalue_input)


class DummyExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(DummyExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: Union[th.Tensor, TensorDict]) -> Union[th.Tensor, TensorDict]:
        return observations


class CGatePolicy(TD3Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = DummyExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        context_branch_width: int = 256,
        head_width: int = 256,
        **kwargs,
    ):
        super(TD3Policy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "context_branch_width": context_branch_width,
            "head_width": head_width

        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CGateActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CGateCritic(**critic_kwargs).to(self.device)


def get_cgate_policy(agent_name: str) -> Type[BasePolicy]:
    valid_agents = ["TD3", "DDPG"]
    if agent_name not in valid_agents:
        raise ValueError(f"Can only use cGate for {valid_agents}. Requested {agent_name}.")
    return CGatePolicy

