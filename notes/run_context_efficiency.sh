echo "Start Training"
echo "-----------------------------------------"
export env_name=CARLPendulum
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=context_efficiency' 'context_sampler.n_samples=1,2,4,8,16,32,64,128,256,512' '+context_visibility=hidden' --snap_dir ./runs/context_efficiency/${env_name}/train -m


echo "Evaluate on Train Contexts"
echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' folder_id=${env_name}/on_train  --result_dir ./runs/context_efficiency/${env_name}/train


echo "Create Test Contexts"
echo "-----------------------------------------"
python experiments/benchmarking/create_context_efficiency_test_contexts.py $env_name
export contexts_path=runs/context_efficiency/${env_name}/contexts_evaluation.json



echo "Evaluate on Test Contexts"
echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' contexts_path=${contexts_path} folder_id=${env_name}/on_test  --result_dir ./runs/context_efficiency/${env_name}/train