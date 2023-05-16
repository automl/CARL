# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train_ppo(config, env, network):
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["minibatch_size"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )

    def train(rng, env_params, network_params, obsv, env_state):
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    if config["track_metrics"]:
                        out = (total_loss, grads)
                    else:
                        out = (None, None)
                    return train_state, out

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = int(config["minibatch_size"] * config["num_minibatches"])
                assert (
                    batch_size == config["num_steps"] * config["num_envs"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["num_minibatches"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, (total_loss, grads) = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, (total_loss, grads)

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, (loss_info, grads) = jax.lax.scan(
                _update_epoch, update_state, None, config["update_epochs"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            if config["track_traj"]:
                out = (metric, loss_info, grads, traj_batch, advantages)
            elif config["track_metrics"]:
                out = (metric, loss_info, grads, advantages)
            else:
                out = metric
            return runner_state, out

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, out = jax.lax.scan(
            _update_step, runner_state, None, config["num_updates"]
        )
        return runner_state, out

    return train

import jax
import gymnax
import numpy as np
import jax.numpy as jnp
from typing import Sequence
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import gymnasium as gym
from gymnasium.wrappers import AutoResetWrapper, FlattenObservation
import chex
from typing import Union
from gymnax.environments import EnvState, EnvParams

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def ppo(cfg, env, eval_env):
    #env, env_params = make_env(cfg)
    env = AutoResetWrapper(env)
    env = FlattenObservation(env)
    env = GymToGymnaxWrapper(env)
    env_params = None
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, cfg["num_envs"])
    last_obsv, last_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    network = ActorCritic(env.action_space(env_params).n,
            activation=cfg["activation"],
            hidden_size=cfg["hidden_size"],
        )
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    _, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_x)
    #eval_func = make_eval(cfg, network)
    train_func = jax.jit(make_train_ppo(cfg, env, network))
    runner_state, metrics = train_func(
            rng,
            env_params,
            network_params,
            last_obsv,
            last_env_state,)
    

def make_env(instance):
    if instance["env_framework"] == "gymnax":
        env, env_params = gymnax.make(instance["env_name"])
        env = FlattenObservationWrapper(env)
    else:
        env = gym.make(instance["env_name"])
        # Gymnax does autoreset anyway
        env = AutoResetWrapper(env)
        env = FlattenObservation(env)
        env = GymToGymnaxWrapper(env)
        env_params = None
    env = LogWrapper(env)
    return env, env_params


class GymToGymnaxWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env):
        super().__init__()
        self.done = False
        self.env = JaxifyGymOutput(env)
        self.state = None
        self.state_type = None

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ):
        """Environment-specific step transition."""
        # TODO: this is obviously super wasteful for large state spaces - how can we fix it?
        result_shape = jax.core.ShapedArray(
            np.repeat(self.state[None, ...], 3, axis=0).shape, jnp.float32
        )
        result = jax.pure_callback(self.env.step, result_shape, action)
        s = result[0].astype(self.state_type)
        r = result[1].mean()
        d = result[2].mean()
        result_shape = jax.core.ShapedArray((1,), bool)
        self.done = jax.pure_callback(make_bool, result_shape, d)[0]
        return s, {}, r, self.done, {}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        """Environment-specific reset."""
        self.done = False
        self.state, _ = self.env.reset()
        self.state_type = self.state.dtype
        return self.state, {}

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return self.done

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        if isinstance(self.env.action_space, gym.spaces.Box):
            return len(self.env.action_space.low)
        else:
            return self.env.action_space.n

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return self.env.action_space

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            self.env.observation_space.low.shape,
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return gymnax.environments.spaces.Dict({})

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(500)
    
class JaxifyGymOutput(gym.Wrapper):
    def step(self, action):
        s, r, te, tr, _ = self.env.step(action)
        r = np.ones(s.shape) * r
        d = np.ones(s.shape) * int(te or tr)
        return np.stack([s, r, d]).astype(np.float32)


def make_bool(data):
    return np.array([bool(data)])

def to_gymnasium_space(space):
    import gym as old_gym
    if isinstance(space, old_gym.spaces.Box):
        new_space = gym.spaces.Box(low=space.low, high=space.high, dtype=space.low.dtype)
    elif isinstance(space, old_gym.spaces.Discrete):
        new_space = gym.spaces.Discrete(space.n)
    else:
        raise NotImplementedError
    return new_space


class GymToGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = to_gymnasium_space(self.env.action_space)
        self.observation_space = to_gymnasium_space(self.env.observation_space)

    def reset(self, seed=None, options={}):
        return self.env.reset(), {}
    
    def step(self, action):
        s, r, d, i = self.env.step(action)
        return s, r, d, False, i