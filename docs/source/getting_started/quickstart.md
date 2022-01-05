# Quickstart

All CARL environments use [OpenAI's gym interface](https://gym.openai.com/docs/ "Gym Docs") 
for agent-environment interactions. 

To get started using CARL with your own agents, first define
a context set. In this example, we will use the CARLCartPoleEnv with its 
default context and a longer pole. 
```python
from src.envs import CARLCartPoleEnv_defaults as default
longer_pole = default.copy()
longer_pole["pole_length"] = default["pole_length"]*2
contexts = {0: default, 1: longer_pole}
```

Now that we defined a context set, we can use it to create our environment:
```python
from src.envs import CARLCartPoleEnv
env = CARLCartPoleEnv(contexts=contexts)
```
Now you can interact with the environment just like any other gym environment
while the context will change each episode. For a demonstration on what
context can do, see the [example notebook](https://github.com/automl/CARL) in our repository. More
options for environments creation can be found in the [Environments secion](../environments/carl_env.md).