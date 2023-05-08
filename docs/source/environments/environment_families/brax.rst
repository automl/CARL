CARL Brax Environments
######################
In CARL all `Brax locomotion environments <https://github.com/google/brax>`_ are included.
As context features there are external features like gravity or friction or internal features
like joint strength or torso mass.


CARL Ant Environment
***************************
.. image:: ../data/screenshots/ant.png
  :width: 25%
  :align: center
  :alt: Screenshot of CARLAnt


Here the agent needs to learn how to control a four-legged ant in order
to run (quickly) into a given direction.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLAnt.csv
   :header-rows: 1



CARL Fetch Environment
**********************
.. image:: ../data/screenshots/fetch.png
    :width: 25%
    :height: 100px
    :align: center
    :alt: Screenshot of CARLFetch


Fetch trains a robotic dog to run to a target location.
The target radius and distance as well as physical properties can be varied via the context features.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLFetch.csv
   :header-rows: 1



CARL Grasp Environment
**********************
.. image:: ../data/screenshots/grasp.png
    :width: 25%
    :align: center
    :alt: Screenshot of CARLGrasp


In CARL Grasp the agent is trained to pick up an object with a robot hand.
Three bodies are observed by Grasp: 'Hand', 'Object', and 'Target'.
When Object reaches Target, the agent is rewarded.
Apart from Grasp's pyhiscal properties the target radius, height and distance are also varied.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLGrasp.csv
   :header-rows: 1



CARL Halfcheetah Environment
*****************************
.. image:: ../data/screenshots/halfcheetah.png
    :width: 25%
    :align: center
    :alt: Screenshot of CARLHalfcheetah


A Halfcheetah is trained to run in a given direction.
The context features can vary physical properties.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLHalfcheetah.csv
   :header-rows: 1



CARL Humanoid Environment
**************************
.. image:: ../data/screenshots/humanoid.png
    :width: 25%
    :align: center
    :alt: Screenshot of CARLHumanoid


Here, a Humanoid needs to learn how to walk forward.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLHumanoid.csv
   :header-rows: 1


CARL UR5e Environment
**********************
.. image:: ../data/screenshots/ur5e.png
    :width: 25%
    :align: center
    :alt: Screenshot of CARLUr5e


The agent needs to learn how to move a ur5e robot arm and its end effector to a sequence of targets.
The robot arm has 6 joints.


.. csv-table:: Defaults and Bounds
   :file: ../data/context_definitions/CARLUr5e.csv
   :header-rows: 1
