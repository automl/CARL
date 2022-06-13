# Questions To Answer
- What is the effect of instances?
- How do the state distributions look like for different instances?
- calculate similarity based on KL Divergence between state distributions?
- RL agent verification: At which perturbation per feature does the agent fail?


Key ideas:
- context features as HP configuration
- measure similarity of instances via their induced state distributions



When does the agent collapse?
Optimization task: Find the perturbation of the context features resulting in the worst
performance of an agent trained on the default env with BO. Record state distribution.
Correlation between performance and KL divergence of rho_default and rho_instance?
Maybe we see that different agents perform well on different areas of the instance
space (comment from André).



# Prepare scripts

## run_train_optimal_agent.py
- train coax agent on standard task (default instance)
- SAC
- 1M steps
- Pendulum
- n_seeds
    
## run_perturbance_optimization // collect data
Inputs:
- budget: n_eval
- optimal performance
- configuration space = context features with their lower and upper bounds (either one each or all)
- use SMAC to perturb instances (motivation: difficult to cover context feature space)
- perturbation / configuration: difference of instance feature in log space
- record state distributions
- record cumulative rewards
- n_episodes, n_seeds
    
## analyze_perturbance
- visualize state distributions per perturbation
- visualize incumbent over time of perturbance (how long did it take to let the agent fail?)
- report normalized cumulative reward (IQM...), normalized by optimal performance (performance of trained agent on default instance)
- correlation between KL div between state distributions and performance? Hypothesis: the closer the induced state distribution to the default distr, the better the performance
- How to find out state distributions? could also use n_random_agents
- metric for continuous features: AUC. for discrete features: IQM or avg?
