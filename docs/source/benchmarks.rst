.. _benchmarks:

===============
The Benchmarks
===============

.. role:: python(code)
    :language: python

GenRL currently contains the following benchmarks:

#TODO: cite all envs! Readme can be copied here

* Contextual Classic Control:
  Contextual extensions of simple physics simulations
* Brax walker environments:
  Configurable brax locomotion environments
* Contextual Box2D:
  OpenAI's Box2D environments with context
* TOAD-GAN:
  Directed procedural generation for Mario
* RNA Design:
  Designing RNA structures given structure constraints

Our benchmarks are based on OpenAI's gym interface, so they can be used like any other
standard RL environment.
The context are to train on are given upon initialization and can be accessed via the
"contexts" attribute during runtime:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    context = TODO
    env = CARLCartPoleEnv(contexts = context)
    print(env.contexts)

.. automodule:: src.envs
    :members:
    :show-inheritance: