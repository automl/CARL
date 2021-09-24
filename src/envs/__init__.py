#Modular imports
import importlib
import warnings

#Classic control is in gym and thus necessary for the base version to run
from src.envs.classic_control import *

#Environment loading
box2D_spec = importlib.util.find_spec("Box2D")
found = box2d_spec is not None
if found:
    from src.envs.box2d import *
else:
    warnings.warn("Module 'Box2D' not found. If you want to use these environments, please follow the installation guide.")

brax_spec = importlib.util.find_spec("brax")
found = brax_spec is not None
if found:
    from src.envs.brax import *
else:
    warnings.warn("Module 'Brax' not found. If you want to use these environments, please follow the installation guide.")

try:
    from src.envs.mario import *
except:
    warnings.warn("Module 'Mario' not found. Please follow installation guide for ToadGAN environment.")

RNA_spec = importlib.util.find_spec("RNA")
found = RNA_spec is not None
if found:
    from src.envs.rna import *
else:
    warnings.warn("Module 'RNA' not found. Please follow installation guide for RNA environment.")

