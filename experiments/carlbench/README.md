# CARL Benchmark

- `carl` provides the contextual environments
- `carlbench` provides the associated benchmarks with different levels of difficulty.

## Environment to Benchmark

For a proper benchmark for Reinforcement Learning we need more than just the 
environment.
We also need an evaluation protocol with train and test instances as well as 
an aggregation strategy.

### Train and Test Instances

We decide to parametrize distributions from which the instances are drawn.
We select multi-variate Gaussian distributions (although we can specify any 
distribution). 
Instead of providing a fixed set for the train and test instance we define the 
parametrizations of the distribution with a seed and a number of samples.
The mean always is the default context feature value.
The variance determines different levels of difficulty for the distributions:

- easy: `sigma_rel` = 0.1
- medium: `sigma_rel` = 0.25
- hard: `sigma_rel` = 0.5

In addition, we can specifiy the number of samples, either freely or with the presets:
- small: 100 samples
- medium: 1000 samples
- large: 10000 samples

The distributions can be passed via the `context_selector` argument.


### Example Yaml Config 
```yaml
context_sampler:
  _target_: experiments.carlbench.context_sampling.ContextSampler
  env_name: ${env}
  context_feature_names: ${context_feature_names}
  seed: ${seed}
  n_samples: 1000
  sigma_rel: 0.1  # 0.1, 0.25, 0.5
```


### Aggregation Strategy
Depending on which aggregation strategy you select you can emphasize or deemphasize
certain aspects of your algorithm. Therefore we propose to use the aggregation 
strategy of Interquartile Mean (IQM), implemented in the `rliable` package.


## TODOs
- [ ] Implement context selector with different settings and logging
- [ ] Integrate `rliable` and use IQM
- [ ] Build in option to draw new context sample each time