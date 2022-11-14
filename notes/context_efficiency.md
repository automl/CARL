# Context Efficiency
> How many samples of context does the agent need to generalize to the test distribution?

Note: We want to see each context at least twice during training.

## Setup
- agent: SAC
- env: Pendulum
- vary: l ~ U(0.5, 1.5)
- 10 seeds
- hidden
- n contexts=1,2,4,8,16,32,64,128,256,512
- n samples test: 1024
- n training steps: 512 * 2 * 200 = 204800

## Command

```bash
bash notes/run_context_efficiency.sh CARLPendulumEnv
```


### Debug
```bash
export env_name=$1
python experiments/benchmarking/run_training.py 'seed=range(1,2)' '+experiments=context_efficiency' 'context_sampler.n_samples=1' '+context_visibility=hidden' 'max_num_frames=5000' 'wandb.debug=true' --snap_dir ./runs/context_efficiency/${env_name}/train -m

python experiments/benchmarking/create_context_efficiency_test_contexts.py $env_name
export contexts_path=runs/context_efficiency/${env_name}/contexts_evaluation.json

python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' contexts_path=${contexts_path} folder_id=${env_name}/on_test 'n_eval_episodes_per_context=1'  'wandb.debug=true' --result_dir ./runs/context_efficiency/${env_name}/train

python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' folder_id=${env_name}/on_train 'wandb.debug=true' 'n_eval_episodes_per_context=1' --result_dir ./runs/context_efficiency/${env_name}/train
```