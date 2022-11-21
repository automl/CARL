# Optimality Gap
> How big is it?:)

## Setup
- env: CartPole
- agent: c51
- vary: pole_length ~ U(0.5, 1.5)
- n_samples: 64
- 10 seeds
- visibility: hidden

1. Train general agent on 64 contexts
2. Evaluate that one on the train contexts
3. Train 1 agent each on the 64 contexts
4. Evaluate all those

Run
```bash
bash notes/run_optimality_gap.sh env_name
```
