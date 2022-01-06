<img align="left" width="80" src="./docs/source/figures/CARL_logo.png" alt="CARL Logo">

# The CARL Benchmark Library
CARL (context adaptive RL) provides highly configurable contextual extensions
to several well-known RL environments. 
It's designed to test your agent's generalization capabilities
in all scenarios where intra-task generalization is important.

Feel free to check out our [paper](https://arxiv.org/abs/2110.02102) and our short [blog post](https://www.automl.org/carl-a-benchmark-to-study-generalization-in-reinforcement-learning/)!

## Benchmarks
Benchmarks include:
- [OpenAI gym classic control suite](https://gym.openai.com/envs/#classic_control) extended with several physics 
  context features like gravity or friction
    
- [OpenAI gym Box2D](https://gym.openai.com/envs/#box2d) BipedalWalker, LunarLander and
  CarRacing, each with their own modification possibilities like
  new vehicles to race
  
- All [Brax locomotion environments](https://github.com/google/brax) with exposed internal features
  like joint strength or torso mass
  
- [Super Mario (TOAD-GAN)](https://github.com/Mawiszus/TOAD-GAN), a procedurally generated jump'n'run game with control
  over level similarity
  
- [RNADesign](https://github.com/automl/learna/), an environment for RNA design given structure
  constraints with structures from different datasets to choose from

![Screenshot of each environment included in CARL.](./docs/source/figures/envs_overview.png)
   

## Installation
We recommend you use a virtual environment (e.g. Anaconda) to 
install CARL and its dependencies. We recommend and test with python 3.9 under Linux.

First, clone our repository and install the basic requirements:
```bash
git clone https://github.com/automl/CARL.git --recursive
cd CARL
pip install .
```
This will only install the basic classic control environments, which should run on most operating systems. For the full set of environments, use the install options:
```bash
pip install -e .[box2d, brax, rna, mario]
```
These may not be compatible with Windows systems. Box2D environment may need to be installed via conda on MacOS systems:
```bash
conda install -c conda-forge gym-box2d
```
In general, we test on Linux systems, but aim to keep the benchmark compatible with MacOS as much as possible. 
Mario at this point, however, will not run on any operation system besides Linux

To install the additional requirements for ToadGAN:
```bash
javac src/envs/mario/Mario-AI-Framework/**/*.java
```

If you want to use the RNA design environment:
```bash
cd src/envs/rna/learna
make requirements
make data
```
In case you want to run our experiments or use our training files, also install the experiment dependencies:
```bash
pip install -e .[experiments]
```
## Train an Agent
To get started with CARL, you can use our 'train.py' script.
It will train a PPO agent on the environment of your choice
with custom context variations that are sampled from a standard 
deviation. 

To use MetaCartPole with variations in gravity and friction by 20% 
compared to the default, run:
```bash
python train.py 
--env CARLCartPoleEnv 
--context_args gravity friction
--default_sample_std_percentage 0.2
--outdir <result_location>
```
You can use the plotting scripts in src/eval to view the results.

## CARL's Contextual Extension
CARL contextually extends the environment by making the context visible and configurable. During training we therefore can encounter different contexts and train for generalization. We exemplarily show how Brax' Fetch is extended and embedded by CARL. Different instiations can be achieved by setting the context features to different values. 

![CARL contextually extends Brax' Fetch.](./docs/source/figures/concept.png)

## Cite Us
If you use CARL in your research, please cite our paper on the benchmark:

```bibtex
@inproceedings { BenEim2021a,
  author = {Carolin Benjamins and 
            Theresa Eimer and 
            Frederik Schubert and 
            André Biedenkapp and 
            Bodo Rosenhahn and Frank Hutter and 
            Marius Lindauer},
  title = {CARL: A Benchmark for Contextual and Adaptive Reinforcement Learning},
  booktitle = {NeurIPS 2021 Workshop on Ecological Theory of Reinforcement Learning},
  year = {2021},
  month = dec
}
```

## References
[OpenAI gym, Brockman et al., 2016. arXiv preprint arXiv:1606.01540](https://arxiv.org/pdf/1606.01540.pdf)

[Brax -- A Differentiable Physics Engine for Large Scale 
Rigid Body Simulation, Freeman et al., NeurIPS 2021 (Dataset & 
Benchmarking Track)](https://arxiv.org/pdf/2106.13281.pdf)

[TOAD-GAN: Coherent Style Level Generation from a Single Example,
Awiszus et al., AIIDE 2020](https://arxiv.org/pdf/2008.01531.pdf)

[Learning to Design RNA, Runge et al., ICRL 2019](https://arxiv.org/pdf/1812.11951.pdf)

## License
CARL falls under the Apache License 2.0 (see file 'LICENSE') as is permitted by all 
work that we use. This includes CARLMario, which is not based on the Nintendo Game, but on
TOAD-GAN and TOAD-GUI running under an MIT license. They in turn make use of the Mario AI framework
(https://github.com/amidos2006/Mario-AI-Framework). This is not the original game but a replica, 
explicitly built for research purposes and includes a copyright notice (https://github.com/amidos2006/Mario-AI-Framework#copyrights ). 
