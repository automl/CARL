The CARL Environment
====================

CARL extends the standard `gym interface <https://gymnasium.farama.org/content/basic_usage/>`_  with context.
This context changes the environment's transition dynamics and reward
function, creating a greater challenge for the agent.
During training we therefore can encounter different contexts and train for generalization.
We exemplarily show how Brax' Fetch is extended and embedded by CARL. Different
instantiations can be achieved by setting the context features to different values.

.. image:: ../figures/concept.png
  :width: 75%
  :align: center
  :alt: CARL contextually extends Brax' Fetch.


Here we give a brief overview of the available options on how to create
and work with contexts.

A context takes the form of a dictionary with a key-value pair for each
context feature. An example is the default context for the CARLAnt environment:

.. code-block:: python

    Ant_defaults = {
        "joint_stiffness": 5000,
        "gravity": -9.8,
        "friction": 0.6,
        "angular_damping": -0.05,
        "actuator_strength": 300,
        "joint_angular_damping": 35,
        "torso_mass": 10,
    }

The context set used for training is comprised of at least one context.
It is also a dictionary with keys that should identify the context in a
meaningful way, e.g. an id. An example of a simple instance set for 
CARLCartPole would be:
    
.. code-block:: python

    from src.envs import CARLCartPoleEnv_defaults as default
    longer_pole = default.copy()
    longer_pole["pole_length"] = default["pole_length"]*2
    contexts = {0: default, 1: longer_pole}


This context set can then be used to create the environment:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    env = CARLCartPoleEnv(contexts=contexts)

Per default, the context will be changed each episode in a round robin
fashion. 

The user can choose if and how the context is provided to the agent.
If the context should be hidden completely, instantiate the environment
with the 'hide_context' option:


.. code-block:: python

    from src.envs import CARLCartPoleEnv
    env = CARLCartPoleEnv(contexts=contexts, hide_context=True)


By default, the context is visible and concatenated onto the state information
with no separation between state and context features. They can be 
*provided separately*, though, using dict observations:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    env = CARLCartPoleEnv(contexts=contexts, hide_context=False, dict_observation_space=True)


Furthermore, users can choose to provide the full context information (default)
or only a *subset*:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    env = CARLCartPoleEnv(contexts=contexts, state_context_features=["gravity", "pole_length"])


Context features can also be normalized or augmented with noise to either
make learning easier or more difficult.
In addition, only the context features *changing across the contexts* provided can be appended to the state like so:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    env = CARLCartPoleEnv(contexts=contexts, state_context_features="changing_context_features")

