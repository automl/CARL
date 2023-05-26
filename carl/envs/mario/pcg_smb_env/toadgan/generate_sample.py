# Code from https://github.com/Mawiszus/TOAD-GAN
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


# Generates a noise tensor. Uses torch.randn.
def generate_spatial_noise(
    size: Union[Any, List[int], Tuple[int]],
    device: Union[str, torch.device] = "cpu",
    seed: int = 0,
) -> Tensor:
    return torch.randn(
        size,
        device=device,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    )


# Generate a sample given a TOAD-GAN and additional parameters
@torch.no_grad()  # type: ignore [misc]
def generate_sample(
    generators: List[nn.Module],
    noise_maps: Tensor,
    reals: Tensor,
    noise_amplitudes: Tensor,
    num_layer: int,
    token_list: Tensor,
    scale_v: float = 1.0,
    scale_h: float = 1.0,
    current_scale: int = 0,
    gen_start_scale: int = 0,
    initial_noise: Optional[Tensor] = None,
) -> List[str]:
    in_s = None
    images_cur: List[Tensor] = []
    images: List[Tensor] = []
    z_s: List[Tensor] = []

    device = torch.device("cpu")

    # Main loop
    for G, Z_opt, noise_amp in zip(generators, noise_maps, noise_amplitudes):
        if current_scale >= len(generators):
            break  # should not be reached

        # Zero Padding
        n_pad = int(num_layer)
        m = nn.ZeroPad2d(int(n_pad))

        # Calculate actual shape
        nzx = (Z_opt.shape[2] - n_pad * 2) * scale_v
        nzy = (Z_opt.shape[3] - n_pad * 2) * scale_h

        # Init images list
        images_prev = images_cur
        images_cur = []
        channels = len(token_list)

        # Init in_s
        if in_s is None:
            in_s = torch.zeros(reals[0].shape[0], channels, *reals[0].shape[2:]).to(
                device
            )
        elif in_s.sum() == 0:
            in_s = torch.zeros(in_s.shape[0], channels, *in_s.shape[2:]).to(device)

        if current_scale == 0:  # First step: Make base noise
            if initial_noise is not None and len(initial_noise) > 0:
                z_curr = initial_noise.float().to(device)
            else:
                z_curr = generate_spatial_noise(
                    [1, channels, int(round(nzx)), int(round(nzy))], device=device
                )
            z_curr = m(z_curr)
        else:  # All other steps: Make added noise
            if current_scale < gen_start_scale:
                z_curr = z_s[current_scale]
            else:
                z_curr = generate_spatial_noise(
                    [1, channels, int(round(nzx)), int(round(nzy))], device=device
                )
                z_curr = m(z_curr)
        if (not images_prev) or current_scale == 0:  # if there is no "previous" image
            I_prev = in_s
        else:
            I_prev = images[current_scale - 1]

        # Bilinear interpolation for upscaling
        I_prev = interpolate(
            I_prev,
            [int(round(nzx)), int(round(nzy))],
            mode="bilinear",
            align_corners=False,
        )
        I_prev = m(I_prev)

        # Main Step
        z_in = noise_amp * z_curr + I_prev
        I_curr = G(z_in, I_prev, temperature=1)

        # Append results
        images_cur.append(I_curr)

        if current_scale >= gen_start_scale:
            images.append(I_curr)
            z_s.append(z_curr)
        current_scale += 1

    return one_hot_to_ascii_level(I_curr, token_list)


def one_hot_to_ascii_level(level: Any, tokens: Any) -> List[str]:
    """Converts a full token level tensor to an ascii level."""
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level
