"""
This code was adopted from:
https://github.com/AlexGraikos/diffusion_priors
"""

import os
from os.path import join as pjoin

import numpy as np
import torch
import math
from tqdm import tqdm
import torchvision.transforms.functional as tvtf
from torchvision.utils import make_grid

import osmosis_utils.utils as utilso


class GaussianDiffusion:
    """

    Gaussian Diffusion process with linear beta scheduling

    """

    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            # Generate an extra alpha for bT
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(0)
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0, t):
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t - 1]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1 - atbar) * epsilon
        return xt, epsilon

    def inverse(self, net, shape=(1, 64, 64), image_channels=3, steps=None, x=None, start_t=None, device='cpu',
                **kwargs):
        # Specify starting conditions and number of steps to run for

        record_process = kwargs.get("record_process", False)
        record_every = kwargs.get("record_every", 200)
        save_path = kwargs.get("save_path", None)
        image_idx = kwargs.get("image_idx", 0)

        if record_process:
            xt_list = []
            x_rgb_list = []
            x_depth_list = []

        if x is None:
            x = torch.randn((1,) + shape).to(device)
        if start_t is None:
            start_t = self.T
        if steps is None:
            steps = self.T

        for t in tqdm(range(start_t, start_t - steps, -1)):

            at = self.alpha[t - 1]
            atbar = self.alphabar[t - 1]

            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t - 2]
                beta_tilde = self.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

            else:
                z = torch.zeros_like(x)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x, t.float().to(device))[:, :image_channels, :, :]

            if record_process and (save_path is not None) and ((not t.item() % record_every) or (t.item() == 1)):
                t_vis = t.detach().cpu().item()

                # save noised image - xt
                x_vis = torch.clamp(0.5 * (x.detach().cpu().squeeze() + 1), 0, 1)
                # tvtf.to_pil_image(x_vis).save(os.path.join(save_path, f"process_{t_vis}.png"))
                xt_list.append(x_vis[0:3])

                # save predicted image - x_0_hat (x_start_pred)
                x_start_pred = (1 / np.sqrt(atbar)) * (
                        x.detach().cpu() - (np.sqrt(1 - atbar) * pred.detach().cpu()))
                x_start_pred_vis = 0.5 * (x_start_pred.squeeze() + 1)
                # tvtf.to_pil_image(x_start_pred_vis).save(os.path.join(save_path, f"process_pred_{t_vis}.png"))

                x_start_rgb = torch.clamp(x_start_pred_vis[0:3], 0, 1)
                x_rgb_list.append(x_start_rgb)

                # in the rgbd case
                if image_channels == 4:
                    x_depth = utilso.min_max_norm_range_percentile(x_start_pred_vis[3].unsqueeze(0),
                                                                   percent_low=0.05, percent_high=0.99)
                    x_depth = utilso.depth_tensor_to_color_image(x_depth)
                    x_depth_list.append(x_depth)

            x = (1 / np.sqrt(at)) * (x - ((1 - at) / np.sqrt(1 - atbar)) * pred) + np.sqrt(beta_tilde) * z

        # save the grid images of the process
        if record_process and (save_path is not None):
            grid_list = xt_list + x_rgb_list + x_depth_list
            grid_image = make_grid(grid_list, nrow=len(xt_list), pad_value=1.)
            tvtf.to_pil_image(grid_image).save(pjoin(save_path, f"image_{image_idx}_process.png"))

        return x
