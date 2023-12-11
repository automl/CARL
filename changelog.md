# 1.1.0
- Add gymnax classic control environments (#90)

# 1.0.0
Major overhaul of the CARL environment
- Contexts are stored in each environment's class
- Removed deprecate code from CARL env
- CARL env always returns a dict observation, with `obs` and `context`
- Introduction of the context space from which we can conveniently define sampling distributions and sample

Other
- Update brax environments
- Add docs for dmc environments

# 0.2.2
- Make sampling of contexts deterministic with seed
- Update gym to gymnasium

# 0.2.1
- Add Finger (DMC) env
- Readd RNA env (#78)

# 0.2.0
- Integrate dm control environments (#55)
- Add context masks to only append those to the state (#54)
- Extend classic control environments to parametrize initial state distributions (#52)
- Remove RNA environment for maintenance (#61)
- Fixed pre-commit (mypy, black, flake8, isort) (#62)

# 0.1.0
- Initial release.
