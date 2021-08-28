.. _benchmarks:

===============
The Benchmarks
===============

.. role:: python(code)
    :language: python

CARL currently contains the following benchmarks which are contextually extended:

- [OpenAI's classic control suite](https://gym.openai.com/envs/#classic_control) extended with several physics 
  context features like gravity or friction
    
- The OpenAI gym [Box2D environments](https://gym.openai.com/envs/#box2d) BipedalWalker, LunarLander and
  CarRacing, each with their own modification possibilities like
  new vehicles to race
  
- All [Brax locomotion environments](https://github.com/google/brax) with exposed internal features
  like joint strength or torso mass
  
- [ToadGAN](https://github.com/Mawiszus/TOAD-GAN), a procedurally generated jump'n'run game with control
  over level similarity
  
- [RNADesign](https://github.com/automl/learna/), an environment for RNA design given structure
  constraints with structures from different datasets to choose from
  
.. image:: figures/envs_overview.png
  :width: 400
  :alt: Screenshot of each environment included in CARL.

Our benchmarks are based on OpenAI's gym interface, so they can be used like any other
standard RL environment.
The train contexts are passed upon initialization of the CARL environment and can be accessed via the
"contexts" attribute during runtime:

.. code-block:: python

    from src.envs import CARLCartPoleEnv
    from src.context_sampler import sample_contexts
    
    contexts = sample_contexts(
        env_name="CARLCartPoleEnv",  # we need to know the env in order to know the allowed sampling ranges
        context_feature_args=["gravity", "masscart"],  # let's only vary gravity and masscart
        num_contexts=10,  
        default_sample_std_percentage=0.05, # sample values from a normal distribution with mean=default value and std=0.05*default
    )
    env = CARLCartPoleEnv(contexts = contexts)
    print(env.contexts)

.. automodule:: src.envs
    :members:
    :show-inheritance:
    
 
   
