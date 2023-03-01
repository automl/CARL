env_name=$1

if [ -z "$env_name" ]
then
    echo "env_name is not set. Exit."
    exit 1
fi

overrides=""

if [ "$env_name" = "CARLCartPoleEnv" ]; then
    overrides="" 
elif [ "$env_name" = "CARLAcrobotEnv" ]; then
    overrides=" '+environments/classic_control=acrobot' '+context_sampler.context_feature_names=[link_length_1,link_length_2]' '+context_sampler.uniform_bounds_rel=[0.75,1.25]' "
elif [ "$env_name" = "CARLPendulumEnv" ]; then
    overrides=" '+experiments/optgap/pendulum' "
elif [ "$env_name" = "CARLDmcWalkerEnv" ]; then
    overrides=" +experiments/optgap=walker +slurm=cpu carl.context_mask=[] "
fi

echo $env_name
echo $overrides

# echo "Train General Agent"
# echo "-----------------------------------------"
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=optimality_gap' $overrides --snap_dir ./runs/optimality_gap/${env_name}/train_general -m

# echo "Evaluate General Agent on Train Contexts"
# echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=optimality_gap' folder_id=${env_name}/eval_general  --result_dir ./runs/optimality_gap/${env_name}/train_general -m


# echo "Create Runcommands to Train Oracle"
# echo "-----------------------------------------"
python experiments/benchmarking/create_optimality_gap_runcommands_oracle.py $env_name $overrides


# echo "Train Oracle"
# echo "-----------------------------------------"
bash ./runs/optimality_gap/${env_name}/runcommands/run_train_oracles.sh


# echo "Evaluate Oracles on Train Contexts"
# echo "-----------------------------------------"
python experiments/evaluation/run_evaluation.py '+experiments=optimality_gap' folder_id=${env_name}/eval_oracle  --result_dir ./runs/optimality_gap/${env_name}/train_oracle -m
