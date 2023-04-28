Environment Families
====================
In CARL we provide several contextually extended environments where we can set and augment their contexts.
With this we can analyze how the context and its augmentation influences the agent's generalization capabilities,
learning and behavior.

.. image:: ../../figures/envs_overview.png
  :width: 90%
  :align: center
  :alt: Screenshot of each environment included in CARL.

Although each environment has different tasks, goals and mechanics, the behavior of the dynamics and the rewards is
influenced by physical properties.

CARL currently contains the following benchmarks which are contextually extended:


Classic Control
---------------
`OpenAI gym classic control suite <https://gymnasium.farama.org/environments/classic_control/>`_ extended with several physics
context features like gravity or friction

.. toctree::
    :maxdepth: 2

    classic_control

Box2D
-----
`OpenAI gym Box2D <https://gymnasium.farama.org/environments/box2d/>`_ BipedalWalker, LunarLander and
CarRacing, each with their own modification possibilities like
new vehicles to race

.. toctree::
    :maxdepth: 2

    box2d

Brax
----
All `Brax locomotion environments <https://github.com/google/brax>`_ with exposed internal features
like joint strength or torso mass

.. toctree::
    :maxdepth: 2

    brax

Mario
-----
`Super Mario (TOAD-GAN) <https://github.com/Mawiszus/TOAD-GAN>`_, a procedurally generated jump'n'run game with control
over level similarity

.. toctree::
    :maxdepth: 2

    toad_gan

RNA Design
----------
`RNADesign <https://github.com/automl/learna/>`_, an environment for RNA design given structure
constraints with structures from different datasets to choose from

.. toctree::
    :maxdepth: 2

    rna


Overview Table
--------------
Below every environment is listed with its number of context features and the type of action and observation space.

.. csv-table::
   :file: ../data/tab_overview_environments.csv
   :header-rows: 1
