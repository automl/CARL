# cRL_HPO
This is the companion repo for out EcoRL workshop Paper 'Hyperparameters in Contextual RL are Highly Situational'. For our results and insights, please check out the paper itself.

## Installation
If you do not have a virtual environment, we recommend to create and activate a virtual environment (e.g. Anaconda) first.
We recommend to use python 3.9 under Linux.

Then, install CARL (https://github.com/automl/CARL#installation) in your workdir:
```
git clone https://github.com/automl/CARL.git --recursive
cd CARL
pip install gym[box2d]
pip install -e .
```

Now, please clone this repo and install.
Requirements:
ray[tune]
stable_baselines3
sb3-contrib

## Re-running the Optimization
To redo our PB2 runs, call cRL_HPO/run_pb2.py with your chosen arguments. An example for Acrobot:
```
python cRL_HPO/run_pb2.py --name acrobot_hidden --hide_context --outdir results --env CARLAcrobotEnv --context_feature link_length_1 
```

## Evaluating found schedules
We provide the hyperparameter schedules found in our experiments in schedules/{env_name}. To rerun them, use (e.g. for Acrobot):
```
python cRL_HPO/play_pb2.py --outdir evaluation_acrobot_hidden --env CARLAcrobotEnv --context_feature link_length_1 --policy_path <path_to_hp_policy> --seed <seed>
```
