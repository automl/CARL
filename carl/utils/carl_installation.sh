# in .
# installation for HP_opt
# if on LUIS cluster: check if installation succeeds. might fail to limited memory quota
# linux, conda environment

# carl packages
# code breaks when using newest stable_baselines3 so stick to the package versions!
pip install -e .[box2d, brax, experiments]
conda install -c conda-forge gym-box2d

# smac
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac

