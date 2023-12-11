import os
import socket
from contextlib import closing

from jdk4py import JAVA
from py4j.java_gateway import JavaGateway

MARIO_AI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Mario-AI-Framework")
)


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def load_level(level_name: str = "lvl-1.txt"):
    prefix = (
        os.path.join(MARIO_AI_PATH, "levels", "original")
        if level_name.startswith("lvl-")
        else ""
    )
    with open(os.path.join(prefix, level_name), "r") as f:
        level = f.read()
    return level


def launch_gateway():
    free_port = find_free_port()
    return (
        JavaGateway.launch_gateway(
            classpath=os.path.join(
                MARIO_AI_PATH, "Mario-AI-Framework-0.8.0-SNAPSHOT.jar"
            ),
            die_on_exit=True,
            port=free_port,
            java_path=str(JAVA),
            javaopts=["-Djava.awt.headless=false"],
        ),
        free_port,
    )
