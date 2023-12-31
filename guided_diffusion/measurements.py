'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import numpy as np
from torch.nn import functional as F
from torchvision import torch
# from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

import osmosis_utils.utils as utilso

# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls

        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")

    operator = __OPERATOR__[name](**kwargs)
    operator.__name__ = name

    # return __OPERATOR__[name](**kwargs)
    return operator


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size

    def forward(self, data, **kargs):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


# osmosis - learnable Operator
class LearnableOperator(ABC):

    @abstractmethod
    def forward(self, data, **kwargs):
        pass


@register_operator(name='haze_physical')
class HazePhysicalOperator(LearnableOperator):
    def __init__(self, device, beta, b_inf, beta_eta=1, b_inf_eta=1,
                 beta_learn_flag=True, b_inf_learn_flag=True,
                 batch_size=1, **kwargs):
        self.device = device
        self.degamma = kwargs.get("degmma", False)
        self.depth_type = kwargs.get("depth_type", None)
        tmp_value = kwargs.get("value", None)
        self.value = utilso.get_depth_value(tmp_value)

        # initialization values
        # self.beta = torch.tensor(np.fromstring(beta, dtype=float, sep=','), dtype=torch.float, device=device)
        self.beta = torch.tensor(float(beta)).to(device)
        # changing the dimensions for future multiplication
        self.beta = self.beta.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.b_inf = torch.tensor(np.fromstring(b_inf, dtype=float, sep=','), dtype=torch.float, device=device)
        # self.b_inf = torch.tensor(float(b_inf)).to(device)
        self.b_inf = self.b_inf.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.beta_learn_flag = beta_learn_flag
        self.b_inf_learn_flag = b_inf_learn_flag

        # coefficients for the Gradient descend step size
        self.beta_eta = float(beta_eta) if beta_learn_flag else float(0)
        self.b_inf_eta = float(b_inf_eta) if b_inf_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.beta, "lr": self.beta_eta},
                                                                {'params': self.b_inf, "lr": self.b_inf_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 400, 'gamma': 0.0, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)
        if self.degamma:
            rgb_norm = torch.pow(rgb_norm, 2.2)

        depth_tmp = data[:, -1, :, :].unsqueeze(1)

        # convert depth to relevant coordinates
        depth = utilso.convert_depth(depth=depth_tmp, depth_type=self.depth_type, value=self.value)

        # the underwater image formation model
        uw_image = rgb_norm * torch.exp(-self.beta * depth) + self.b_inf * (1 - torch.exp(-self.beta * depth))

        # mask for optimize according closer areas and not far areas
        mask = torch.ones_like(uw_image, device=uw_image.device)
        mask = torch.where((self.beta.detach() * depth.detach()) > 2.5, 0, mask)

        return uw_image, mask

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # update only part of the variables - in this case: self.optimizer == "GD"
        update_beta = self.beta.requires_grad
        update_b_inf = self.b_inf.requires_grad

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                # classic gradient descend
                with torch.no_grad():
                    if update_beta:
                        self.beta.add_(self.beta_a.grad, alpha=-self.beta_eta)
                    if update_b_inf:
                        self.b_inf.add_(self.b_inf.grad, alpha=-self.b_inf_eta)
                # zero the gradients so they will not accumulate
                if update_beta:
                    self.beta.grad.zero_()
                if update_b_inf:
                    self.b_inf.grad.zero_()

            # optimizer was specified
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # return self.beta.detach(), self.b_inf.detach()
        return {'beta': self.beta.detach(), 'b_inf': self.b_inf.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"beta": self.beta.requires_grad,
                            "b_inf": self.b_inf.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.beta.requires_grad_(value["beta"])
            self.b_inf.requires_grad_(value["b_inf"])
        else:
            self.beta.requires_grad_(value)
            self.b_inf.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.beta, self.b_inf]


@register_operator(name='underwater_physical')
class UnderWaterPhysicalOperator(LearnableOperator):
    def __init__(self, device, phi_ab, phi_inf, phi_ab_eta=1, phi_inf_eta=1,
                 phi_ab_learn_flag=True, phi_inf_learn_flag=True,
                 batch_size=1, **kwargs):

        self.device = device
        self.degamma = kwargs.get("degmma", False)

        self.depth_type = kwargs.get("depth_type", None)
        tmp_value = kwargs.get("value", None)
        self.value = utilso.get_depth_value(tmp_value)

        # initialization values
        self.phi_ab = torch.tensor(np.fromstring(phi_ab, dtype=float, sep=','), dtype=torch.float, device=device)
        self.phi_ab = self.phi_ab.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.phi_inf = torch.tensor(np.fromstring(phi_inf, dtype=float, sep=','), dtype=torch.float, device=device)
        self.phi_inf = self.phi_inf.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.phi_ab_learn_flag = phi_ab_learn_flag
        self.phi_inf_learn_flag = phi_inf_learn_flag

        # coefficients for the Gradient descend step size
        self.phi_ab_eta = float(phi_ab_eta) if phi_ab_learn_flag else float(0)
        self.phi_inf_eta = float(phi_inf_eta) if phi_inf_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.phi_ab, "lr": self.phi_ab_eta},
                                                                {'params': self.phi_inf, "lr": self.phi_inf_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 400, 'gamma': 0.0, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)
        if self.degamma:
            rgb_norm = torch.pow(rgb_norm, 2.2)

        depth_tmp = data[:, -1, :, :].unsqueeze(1)

        # convert depth to relevant coordinates
        depth = utilso.convert_depth(depth=depth_tmp, depth_type=self.depth_type, value=self.value)

        # the underwater image formation model
        uw_image = rgb_norm * torch.exp(-self.phi_ab * depth) + self.phi_inf * (1 - torch.exp(-self.phi_ab * depth))

        # mask for optimize according closer areas and not far areas
        mask = torch.ones_like(uw_image, device=uw_image.device)
        mask = torch.where((self.phi_ab.detach() * depth.detach()) > 2.5, 0, mask)

        return uw_image, mask

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # update only part of the variables - in this case: self.optimizer == "GD"
        update_phi_ab = self.phi_ab.requires_grad
        update_phi_inf = self.phi_inf.requires_grad

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                # classic gradient descend
                with torch.no_grad():
                    if update_phi_ab:
                        self.phi_ab.add_(self.phi_ab_a.grad, alpha=-self.phi_ab_eta)
                    if update_phi_inf:
                        self.phi_inf.add_(self.phi_inf.grad, alpha=-self.phi_inf_eta)
                # zero the gradients so they will not accumulate
                if update_phi_ab:
                    self.phi_ab.grad.zero_()
                if update_phi_inf:
                    self.phi_inf.grad.zero_()

            # optimizer was specified
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # return self.phi_ab.detach(), self.phi_inf.detach()
        return {'phi_ab': self.phi_ab.detach(), 'phi_inf': self.phi_inf.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"phi_ab": self.phi_ab.requires_grad,
                            "phi_inf": self.phi_inf.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.phi_ab.requires_grad_(value["phi_ab"])
            self.phi_inf.requires_grad_(value["phi_inf"])
        else:
            self.phi_ab.requires_grad_(value)
            self.phi_inf.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.phi_ab, self.phi_inf]


@register_operator(name='underwater_physical_revised')
class UnderWaterPhysicalRevisedOperator(LearnableOperator):
    def __init__(self, device, phi_a, phi_b, phi_inf,
                 phi_a_eta=1, phi_b_eta=1, phi_inf_eta=1,
                 phi_a_learn_flag=True, phi_b_learn_flag=True, phi_inf_learn_flag=True,
                 batch_size=1, **kwargs):

        self.device = device
        self.degamma = kwargs.get("degamma", False)

        self.depth_type = kwargs.get("depth_type", None)
        tmp_value = kwargs.get("value", None)
        self.value = utilso.get_depth_value(tmp_value)

        # initialization values
        self.phi_a = torch.tensor(np.fromstring(phi_a, dtype=float, sep=','), dtype=torch.float, device=device)
        self.phi_a = self.phi_a.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.phi_b = torch.tensor(np.fromstring(phi_b, dtype=float, sep=','), dtype=torch.float, device=device)
        self.phi_b = self.phi_b.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.phi_inf = torch.tensor(np.fromstring(phi_inf, dtype=float, sep=','), dtype=torch.float, device=device)
        self.phi_inf = self.phi_inf.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        # learning flags
        self.phi_a_learn_flag = phi_a_learn_flag
        self.phi_b_learn_flag = phi_b_learn_flag
        self.phi_inf_learn_flag = phi_inf_learn_flag

        # coefficients for the Gradient descend step size
        self.phi_a_eta = float(phi_a_eta) if phi_a_learn_flag else float(0)
        self.phi_b_eta = float(phi_b_eta) if phi_b_learn_flag else float(0)
        self.phi_inf_eta = float(phi_inf_eta) if phi_inf_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.phi_a, "lr": self.phi_a_eta},
                                                                {'params': self.phi_b, "lr": self.phi_b_eta},
                                                                {'params': self.phi_inf, "lr": self.phi_inf_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 400, 'gamma': 0.0, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)

        if self.degamma:
            rgb_norm = torch.pow(rgb_norm, 2.2)

        depth_tmp = data[:, -1, :, :].unsqueeze(1)

        # convert depth to relevant coordinates
        depth = utilso.convert_depth(depth=depth_tmp, depth_type=self.depth_type, value=self.value)

        # the underwater image formation model
        uw_image = rgb_norm * torch.exp(-self.phi_a * depth) + self.phi_inf * (1 - torch.exp(-self.phi_b * depth))

        return uw_image

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # update only part of the variables - in this case: self.optimizer == "GD"
        update_phi_a = self.phi_a.requires_grad
        update_phi_b = self.phi_b.requires_grad
        update_phi_inf = self.phi_inf.requires_grad

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                # classic gradient descend
                with torch.no_grad():
                    if update_phi_a:
                        self.phi_a.add_(self.phi_a.grad, alpha=-self.phi_a_eta)
                    if update_phi_b:
                        self.phi_b.add_(self.phi_b.grad, alpha=-self.phi_b_eta)
                    if update_phi_inf:
                        self.phi_inf.add_(self.phi_inf.grad, alpha=-self.phi_inf_eta)

                # zero the gradients so they will not accumulate
                if update_phi_a:
                    self.phi_a.grad.zero_()
                if update_phi_b:
                    self.phi_b.grad.zero_()
                if update_phi_inf:
                    self.phi_inf.grad.zero_()

            else:

                self.optimizer.step()
                self.optimizer.zero_grad()

        return {'phi_a': self.phi_a.detach(), 'phi_b': self.phi_b.detach(), 'phi_inf': self.phi_inf.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"phi_a": self.phi_a.requires_grad,
                            "phi_b": self.phi_b.requires_grad,
                            "phi_inf": self.phi_inf.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.phi_a.requires_grad_(value["phi_a"])
            self.phi_b.requires_grad_(value["phi_b"])
            self.phi_inf.requires_grad_(value["phi_inf"])
        else:
            self.phi_a.requires_grad_(value)
            self.phi_b.requires_grad_(value)
            self.phi_inf.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.phi_a, self.phi_b, self.phi_inf]


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: fix the addional Poission noise - osmosis_utils - adaption for debka

        # version 3 (stack-overflow)

        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)
