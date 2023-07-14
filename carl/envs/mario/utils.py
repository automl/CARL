from typing import Tuple

import atexit
import os
import socket
import sys
from contextlib import closing

from py4j.java_gateway import JavaGateway
from xvfbwrapper import Xvfb

MARIO_AI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Mario-AI-Framework")
)
_gateway = None
_port = None


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def load_level(level_name: str = "lvl-1.txt") -> str:
    prefix = (
        os.path.join(MARIO_AI_PATH, "levels", "original")
        if level_name.startswith("lvl-")
        else ""
    )
    with open(os.path.join(prefix, level_name), "r") as f:
        level = f.read()
    return level


def get_port() -> int:
    global _gateway
    global _port
    if _gateway is None:
        _gateway, _port = launch_gateway()
    return _port


def launch_gateway() -> Tuple[JavaGateway, int]:
    vdisplay = Xvfb(width=1280, height=740, colordepth=16)
    vdisplay.start()
    atexit.register(lambda: vdisplay.stop())
    free_port = find_free_port()
    return (
        JavaGateway.launch_gateway(
            classpath=os.path.join(MARIO_AI_PATH, "carl"),
            redirect_stderr=sys.stderr,
            redirect_stdout=sys.stdout,
            die_on_exit=True,
            port=free_port,
        ),
        free_port,
    )
