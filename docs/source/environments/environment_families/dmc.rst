CARL DMC Environments
######################
CARL includes the Finger, Fish, Quadruped and Walker environments from the `DeepMind Control Suite <https://github.com/deepmind/dm_control>`_.
The context features control the MuJoCo physics engine, e.g. the floor friction.


CARL DMC Finger Environment
***************************
.. image:: ../data/screenshots/finger.jpg
  :width: 25%
  :align: center
  :alt: Screenshot of CARLDmcFinger


The agent needs to learn to spin an object using the finger.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLDmcFinger.csv
   :header-rows: 1



CARL DMC Fish Environment
**********************
.. image:: ../data/screenshots/fish.jpg
    :width: 25%
    :height: 100px
    :align: center
    :alt: Screenshot of CARLDmcFish


In Fish, the agent needs to swim as a simulated fish.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLDmcFish.csv
   :header-rows: 1



CARL DMC Quadruped Environment
**********************
.. image:: ../data/screenshots/quadruped.jpg
    :width: 25%
    :align: center
    :alt: Screenshot of CARLDmcQuadruped

.. image:: ../data/context_generalization_plots/plot_ecdf_CARLDmcQuadrupedEnv.png
    :width: 50%
    :align: right
    :alt: Influence of context settings on an agent trained on the default environment.

:raw-html:`<br />`
The agent's goal is to walk efficiently with the quadruped robot.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLDmcQuadruped.csv
   :header-rows: 1



CARL DMC Walker Environment
*****************************
.. image:: ../data/screenshots/walker.jpg
    :width: 25%
    :align: left
    :alt: Screenshot of CARLDmcWalker

.. image:: ../data/context_generalization_plots/plot_ecdf_CARLDmcWalkerEnv.png
    :width: 50%
    :align: right
    :alt: Influence of context settings on an agent trained on the default environment.

The walker robot is supposed to move forward as fast as possible.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLDmcWalker.csv
   :header-rows: 1