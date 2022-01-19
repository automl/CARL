#!/usr/bin/env bash

python run_context_gating.py 'seed=range(0,5)' '+experiment=glob(*)' '+algorithm=ddpg' '+environment=pendulum' 'contexts.context_feature_args=[],[g],[max_speed],[l],[m],[dt]' 'carl.state_context_features=null,${contexts.context_feature_args}' 'carl.gaussian_noise_std_percentage=0.4' --multirun
