import functools
import os
from dataclasses import dataclass
import sys

import torch
from src.envs.mario.generate_sample import generate_sample


@dataclass
class TOADGAN:
    def __init__(self, Gs, Zs, reals, NoiseAmp, token_list, num_layers):
        self.generators = Gs
        self.noise_maps = Zs
        self.reals = reals
        self.noise_amplitudes = NoiseAmp
        self.token_list = token_list
        self.num_layer = num_layers

    @property
    def original_height(self):
        return self.reals[-1].shape[-2]

    @property
    def original_width(self):
        return self.reals[-1].shape[-1]


GENERATOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "TOAD-GUI", "generators", "v2")
)
GENERATOR_PATHS = sorted(
    os.listdir(GENERATOR_DIR),
    key=lambda name: [int(index) for index in name.replace("TOAD_GAN_", "").split("-")],
)


@functools.lru_cache(maxsize=None)
def load_generator(level_index: int):
    import src.envs.mario.models as models
    sys.modules["models"] = models
    gen_path = os.path.join(GENERATOR_DIR, GENERATOR_PATHS[level_index])
    reals = torch.load(
        "%s/reals.pth" % gen_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    Zs = torch.load(
        "%s/noise_maps.pth" % gen_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    NoiseAmp = torch.load(
        "%s/noise_amplitudes.pth" % gen_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    token_list = torch.load("%s/token_list.pth" % gen_path)
    num_layers = torch.load("%s/num_layer.pth" % gen_path)
    Gs = torch.load(
        "%s/generators.pth" % gen_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return TOADGAN(
        Gs=Gs,
        Zs=Zs,
        reals=reals,
        NoiseAmp=NoiseAmp,
        num_layers=num_layers,
        token_list=token_list,
    )


def generate_level(width: int, height: int, level_index: int):
    toad_gan = load_generator(level_index)
    level = generate_sample(
        **vars(toad_gan),
        scale_h=width / toad_gan.original_width,
        scale_v=height / toad_gan.original_height
    )
    return level
