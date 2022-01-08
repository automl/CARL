import os
import sys
from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

#pyglet.options["headless"] = True
pyglet.options["shadow_window"] = False

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

import math
import numpy as np

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens() #available screens
    config = screen[0].get_best_config() #selecting the first screen
    context = config.create_context(None) #create GL context

    return pyglet.window.Window(width=width, height=height, display=display, config=config, context=context, **kwargs)

from gym.envs.classic_control import rendering

print("get display")
width, height = 600, 400
print(os.environ['DISPLAY'])
display = None
# display = get_display(display)
# window = get_window(width=width, height=height, display=display)

viewer = rendering.Viewer(width, height)