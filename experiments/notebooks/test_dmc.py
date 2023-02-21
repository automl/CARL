import numpy as np
import sys
import copy
import os
sys.path.append("../..")
from carl.envs.dmc.loader import load_dmc_env

random_state = np.random.RandomState(42)

env = load_dmc_env(
    domain_name="walker",
    task_name="run"
)

# Simulate episode with random actions
duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()

while env.physics.data.time < duration:

  action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
  time_step = env.step(action)

  # camera0 = env.physics.render(camera_id=0, height=200, width=200)
#   camera1 = env.physics.render(camera_id=1, height=200, width=200)
#   frames.append(np.hstack((camera0, camera1)))
  rewards.append(time_step.reward)
  observations.append(copy.deepcopy(time_step.observation))
  ticks.append(env.physics.data.time)