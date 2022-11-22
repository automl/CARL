export env_name=$1

if [ -z "$env_name" ]
then
    echo "env_name is not set. Exit."
    exit 1
fi


echo "Create Train and Test Contexts"
echo "-----------------------------------------"
python experiments/benchmarking/create_train_test_contexts.py '+experiments=context_efficiency' output_dir=runs/context_efficiency/$env_name/contexts 'context_sampler.n_samples=1024'
contexts_path_train=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/context_efficiency/$env_name/contexts/contexts_train.json
contexts_path_test=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/context_efficiency/$env_name/contexts/contexts_test.json

echo "Start Training"
echo "-----------------------------------------"
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=context_efficiency' 'context_sampler.n_samples=1,2,4,8,16,32,64,128,256,512' contexts_train_path=${contexts_path_train} '+context_visibility=glob(*)' 'wandb.debug=true' --snap_dir ./runs/context_efficiency/${env_name}/train -m


echo "Evaluate on Train Contexts"
echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' folder_id=${env_name}/eval/on_train  --result_dir ./runs/context_efficiency/${env_name}/train -m


# echo "Create Test Contexts"
# echo "-----------------------------------------"
# python experiments/benchmarking/create_context_efficiency_test_contexts.py $env_name
# export contexts_path=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/context_efficiency/${env_name}/contexts_evaluation.json



echo "Evaluate on Test Contexts"
echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=context_efficiency' contexts_path=${contexts_path_test} folder_id=${env_name}/eval/on_test  --result_dir ./runs/context_efficiency/${env_name}/train -m
