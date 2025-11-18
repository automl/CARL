from __future__ import annotations
from dataclasses import asdict
import time
import traceback

from carl.context.selection import StaticSelector
from carl.envs import CARLBraxHalfcheetah
from gymnasium.wrappers import FlattenObservation, FilterObservation #StepAPICompatibility
import numpy as np


def init_carl(carl_env_fn, contexts=None, obs_context_features=None, hide_context=True, context_selector=None):
    env = carl_env_fn(contexts=contexts, 
                      obs_context_features=obs_context_features, 
                      context_selector=context_selector)
    if hide_context:
        env = FlattenObservation(FilterObservation(env, filter_keys=["obs"]))
    else:
        env = FlattenObservation(FilterObservation(env, filter_keys=["obs", "context"]))
    return env

def main():
    '''Fixing the action sequence and changing the dynamics. Then observing the change in trjectories.'''
    def print_context(eval_env):
        inertia_data = asdict(eval_env.env.unwrapped._env.sys.link.inertia)
        link_names = eval_env.env.unwrapped._env.sys.link_names
        # print(f'link_names = {link_names}')
        link_name = context_labels[0].split("_")[-1]
        # print(f'link_name = {link_name}')
        if link_name in link_names:
            idx = link_names.index(link_name)
            # inertia_data["mass"] = inertia_data["mass"].at[idx]
            print(f'{link_names[idx]} = {inertia_data["mass"][idx]}')
        else:
            print(f'link_name={link_name} not found in link_names={link_names}.')
            
    carl_env_fn = CARLBraxHalfcheetah
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()
    context_labels = ["mass_torso"]
    labels = np.array(context_labels)
    n_samples = 5
    rel_std = 0.25
    
    context_mean = []
    for key in DEFAULT_CONTEXT.keys():
        if key in context_labels:
            context_mean.append(DEFAULT_CONTEXT[key])
    context_labels = ["mass_torso"]
    labels = np.array(context_labels)
    
    context_rel_std = rel_std
    context_std = [abs(mean)*context_rel_std for mean in context_mean]
    
    eval_context_array = np.zeros((len(context_mean), n_samples))
    for i in range(len(context_mean)):
        eval_context_array[i,:] = np.random.normal(context_mean[i], context_std[i], n_samples)
    
    eval_context_dict = {}
    for i in range(eval_context_array.shape[1]):
        eval_context_dict[i] = {0:{key:value for key,value in zip(context_labels, eval_context_array[:,i])}}
    
    action_seq = []
    imax = n_samples
    obs_traj = []
    for i in range(imax):
        contexti = eval_context_dict[i]
        print(f'eval_context_dict[{i}]={eval_context_dict[i]}')
        eval_env = init_carl(carl_env_fn, 
                             contexts=contexti, 
                             obs_context_features=context_labels,
                             hide_context=True,
                             context_selector=StaticSelector
                            )
        obs, _ = eval_env.reset()
        print_context(eval_env)
        done = False
        obs_traj.append([])
        ep_rew = 0.0
        obs_traj[i].append(obs)
        timestep = 0
        while not done:
            if i==0:
                action = eval_env.action_space.sample()
                action_seq.append(action)
            else:
                action = action_seq[timestep]
            obs, rew, term, trunc, _ = eval_env.step(action)
            obs_traj[i].append(obs)
            ep_rew += rew
            done = term or trunc
            timestep += 1
        print(f'len(obs_traj) = {len(obs_traj)}, len(obs_traj[{i}]) = {len(obs_traj[i])}, ep_rew = {ep_rew:.2f}')
        
    obs_traj = np.array(obs_traj)
    print(f'obs_traj.shape = {obs_traj.shape}')
    
    for i in range(1,obs_traj.shape[0]):
        obs_traj[i] = obs_traj[i] - obs_traj[0]
        print(f'sum(obs_traj[{i}]) = {np.sum(np.absolute(obs_traj[i])):.4f}')

if __name__ == "__main__":
    main()