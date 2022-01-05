CARL Classic Control Environments
=================================

Classic Control is a problem suit included in Open AI's gym consisting
of simply physics simulation tasks. Context features here are therefore
also physics-based, e.g. friction, mass or gravity.

CARL Pendulum Environment
-------------------------
.. image:: ../data/screenshots/pendulum.jpeg
    :width: 200px
    :align: center
    :alt: Pendulum Environment
In Pendulum, the agent's task is to swing up an inverted pendulum and
balance it at the top from a random position. The action here is the
direction and amount of force the agent wants to apply to the pendulum.

The available context features in Pendulum as well as their defaults are
as follows:
.. csv-table:: Defaults
   :file: ../data/context_defaults/CARLPendulumEnv.csv
   :header-rows: 1

The context feature values are bounded like this:
.. csv-table:: Bounds
   :file: ../data/context_bounds/CARLPendulumEnv.csv
   :header-rows: 1

CARL CartPole Environment
-------------------------
.. image:: ../data/screenshots/cartpole.jpeg
    :width: 200px
    :align: center
    :alt: CartPole Environment
CartPole, similarly to Pendulum, asks the agent to balance a pole upright, though
this time the agent doesn't directly apply force to the pole but moves a cart on which
the pole ist placed either to the left or the right.

The context features for CartPole are:
.. csv-table:: Defaults
   :file: ../data/context_defaults/CARLCartPoleEnv.csv
   :header-rows: 1

Their values should stay between:
.. csv-table:: Bounds
   :file: ../data/context_bounds/CARLCartPoleEnv.csv
   :header-rows: 1

CARL Acrobot Environment
-------------------------
.. image:: ../data/screenshots/acrobot.jpeg
    :width: 200px
    :align: center
    :alt: Acrobot Environment
Acrobot is another swing-up task with the goal being swinging the end of the lower
of two links up to a given height. The agent accomplishes this by actuating
the joint connecting both links.

The context here is given by:
.. csv-table:: Defaults
   :file: ../data/context_defaults/CARLAcrobotEnv.csv
   :header-rows: 1

The context feature bounds consist of:
.. csv-table:: Bounds
   :file: ../data/context_bounds/CARLAcrobotEnv.csv
   :header-rows: 1

CARL MountainCar Environments
------------------------------
.. image:: ../data/screenshots/mountaincar.jpeg
    :width: 200px
    :align: center
    :alt: MountainCar Environment
The MountainCar environment asks the agent to move a car up a steep slope. In order
to succeed, the agent has to accelerate using the opposite slope. There are two
versions of the environment, a discrete one with only "left" and "right" as actions,
as well as a continuous one.

These are the context features for the discrete case:
.. csv-table:: Defaults
   :file: ../data/context_defaults/CARLMountainCarEnv.csv
   :header-rows: 1

Along with their bounds:
.. csv-table:: Bounds
   :file: ../data/context_bounds/CARLMountainCarEnv.csv
   :header-rows: 1

And for the continuous case:
.. csv-table:: Defaults
   :file: ../data/context_defaults/CARLMountainCarContinuousEnv.csv
   :header-rows: 1

With their respective bounds:
.. csv-table:: Bounds
   :file: ../data/context_bounds/CARLMountainCarContinuousEnv.csv
   :header-rows: 1