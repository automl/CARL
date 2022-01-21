#!/usr/bin/env bash

python run_context_gating.py 'seed=range(0,5)' '+experiment=glob(*)' '+algorithm=td3' '+environment=ant' 'contexts.context_feature_args=[]' 'carl.state_context_features=null,${contexts.context_feature_args}' 'contexts.default_sample_std_percentage=0.1' --multirun
