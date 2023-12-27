'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import numpy as np
from torch.nn import functional as F
from torchvision import torch
# from motionblur.motionblur import Kernel

from dps_pattern.util.resizer import Resizer
from dps_pattern.util.img_utils import Blurkernel, fft2_m

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


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''

    def __init__(self, device):
        self.device = device

    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device

    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        return blur_model

    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


# osmosis_utils - add learnable operater
class LearnableOperator(ABC):

    @abstractmethod
    def forward(self, data, **kwargs):
        pass


@register_operator(name='underwater_physical_old')
class UnderWaterPhysicalOperator(LearnableOperator):
    def __init__(self, device, beta, b_inf, beta_eta=1, b_inf_eta=1, beta_learn_flag=True, b_inf_learn_flag=True,
                 batch_size=1, optimizer=None, depth_type=None, value=None, beta_a_depth=False, **kwargs):
        self.device = device
        self.depth_type = depth_type

        # initialization values
        self.beta = torch.tensor(np.fromstring(beta, dtype=float, sep=','), dtype=torch.float, device=device)
        # changing the dimensions for future multiplication
        self.beta = self.beta.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.b_inf = torch.tensor(np.fromstring(b_inf, dtype=float, sep=','), dtype=torch.float, device=device)
        self.b_inf = self.b_inf.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.beta_learn_flag = beta_learn_flag
        self.b_inf_learn_flag = b_inf_learn_flag

        self.beta.requires_grad_(True)
        self.b_inf.requires_grad_(True)

        self.optimizer = optimizer

        # coefficients for the Gradient descend step size
        self.beta_eta = torch.tensor(float(beta_eta), device=device) if beta_learn_flag else 0
        self.b_inf_eta = torch.tensor(float(b_inf_eta), device=device) if b_inf_learn_flag else 0

        # set optimizer
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.beta, "lr": self.beta_eta},
                                                                {'params': self.b_inf, "lr": self.b_inf_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 450, 'gamma': 0.1, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)
        depth_tmp = data[:, -1, :, :].unsqueeze(1)

        # convert depth to relevant coordinates
        depth = utilso.convert_depth(depth=depth_tmp, depth_type=self.depth_type, value=self.value)

        # the underwater image formation model
        uw_image = rgb_norm * torch.exp(-self.beta * depth) + self.b_inf * (1 - torch.exp(-self.beta * depth))

        return uw_image

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                with torch.no_grad():

                    self.beta.add_(self.beta.grad, alpha=-self.beta_eta)
                    self.b_inf.add_(self.b_inf.grad, alpha=-self.b_inf_eta)

                # zero the gradients so they will not accumulate
                self.beta.grad.zero_()
                self.b_inf.grad.zero_()

            else:

                self.optimizer.step()
                self.optimizer.zero_grad()

        return self.beta.detach(), self.b_inf.detach()


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
    def __init__(self, device, beta, b_inf, beta_eta=1, b_inf_eta=1,
                 beta_learn_flag=True, b_inf_learn_flag=True,
                 batch_size=1, **kwargs):
        self.device = device
        self.degamma = kwargs.get("degmma", False)
        self.depth_type = kwargs.get("depth_type", None)
        tmp_value = kwargs.get("value", None)
        self.value = utilso.get_depth_value(tmp_value)

        # initialization values
        self.beta = torch.tensor(np.fromstring(beta, dtype=float, sep=','), dtype=torch.float, device=device)
        # changing the dimensions for future multiplication
        self.beta = self.beta.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.b_inf = torch.tensor(np.fromstring(b_inf, dtype=float, sep=','), dtype=torch.float, device=device)
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


@register_operator(name='underwater_physical_revised')
class UnderWaterPhysicalRevisedOperator(LearnableOperator):
    def __init__(self, device, beta_a, beta_b, b_inf, beta_a_eta=1, beta_b_eta=1, b_inf_eta=1,
                 beta_a_learn_flag=True, beta_b_learn_flag=True, b_inf_learn_flag=True,
                 batch_size=1, **kwargs):
        self.device = device
        self.degamma = kwargs.get("degamma", False)

        self.depth_type = kwargs.get("depth_type", None)
        tmp_value = kwargs.get("value", None)
        self.value = utilso.get_depth_value(tmp_value)

        # initialization values
        beta_a_depth = kwargs.get("beta_a_depth", False)
        self.beta_a_depth = beta_a_depth
        if self.beta_a_depth:
            # beta_a = a * exp(b*depth) + c * exp(d*depth)
            self.beta_a = 0.05 * torch.ones(batch_size, 3, 4, device=device).unsqueeze(-1).unsqueeze(-1)

            # beta_a = a * depth + b
            # self.beta_a = 0.05 * torch.ones(batch_size, 3, 2, device=device).unsqueeze(-1).unsqueeze(-1)

        else:
            self.beta_a = torch.tensor(np.fromstring(beta_a, dtype=float, sep=','), dtype=torch.float, device=device)
            # changing the dimensions for future multiplication
            self.beta_a = self.beta_a.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.beta_b = torch.tensor(np.fromstring(beta_b, dtype=float, sep=','), dtype=torch.float, device=device)
        # changing the dimensions for future multiplication
        self.beta_b = self.beta_b.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.b_inf = torch.tensor(np.fromstring(b_inf, dtype=float, sep=','), dtype=torch.float, device=device)
        self.b_inf = self.b_inf.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)

        self.beta_a_learn_flag = beta_a_learn_flag
        self.beta_b_learn_flag = beta_b_learn_flag
        self.b_inf_learn_flag = b_inf_learn_flag

        # coefficients for the Gradient descend step size
        self.beta_a_eta = float(beta_a_eta) if beta_a_learn_flag else float(0)
        self.beta_b_eta = float(beta_b_eta) if beta_b_learn_flag else float(0)
        self.b_inf_eta = float(b_inf_eta) if b_inf_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.beta_a, "lr": self.beta_a_eta},
                                                                {'params': self.beta_b, "lr": self.beta_b_eta},
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

        # beta_a as function of depth (derya suggested it in sea-thru)
        if self.beta_a_depth:

            # in that case expand() is identical to repeat()
            depth_a = depth.detach().expand(-1, 3, -1, -1)

            # beta_a = a * exp(b*depth) + c * exp(d*depth)
            beta_a_ext = self.beta_a[:, :, 0] * torch.exp(self.beta_a[:, :, 1] * depth_a) + \
                         self.beta_a[:, :, 2] * torch.exp(self.beta_a[:, :, 3] * depth_a)

            # beta_a = a * depth + b
            # beta_a_ext = self.beta_a[:, :, 0] * depth_a + torch.exp(self.beta_a[:, :, 1] * depth_a)

            uw_image = rgb_norm * torch.exp(-beta_a_ext * depth) + self.b_inf * (1 - torch.exp(-self.beta_b * depth))
            self.beta_a_depth_calc = beta_a_ext.detach()
            mask = None

        else:
            # the underwater image formation model
            uw_image = rgb_norm * torch.exp(-self.beta_a * depth) + self.b_inf * (1 - torch.exp(-self.beta_b * depth))

            # mask for optimize according closer areas and not far areas
            mask = torch.ones_like(uw_image, device=uw_image.device)
            mask = torch.where((self.beta_a.detach() * depth.detach()) > 2.5, 0, mask)

        return uw_image, mask

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # update only part of the variables - in this case: self.optimizer == "GD"
        update_beta_a = self.beta_a.requires_grad
        update_beta_b = self.beta_b.requires_grad
        update_b_inf = self.b_inf.requires_grad

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                # classic gradient descend
                with torch.no_grad():
                    if update_beta_a:
                        self.beta_a.add_(self.beta_a.grad, alpha=-self.beta_a_eta)
                    if update_beta_b:
                        self.beta_b.add_(self.beta_b.grad, alpha=-self.beta_b_eta)
                    if update_b_inf:
                        self.b_inf.add_(self.b_inf.grad, alpha=-self.b_inf_eta)

                # zero the gradients so they will not accumulate
                if update_beta_a:
                    self.beta_a.grad.zero_()
                if update_beta_b:
                    self.beta_b.grad.zero_()
                if update_b_inf:
                    self.b_inf.grad.zero_()

            else:

                self.optimizer.step()
                self.optimizer.zero_grad()

        # if self.beta_a_depth:
        #     return [self.beta_a_depth_calc.detach(), self.beta_b.detach()], self.b_inf.detach()
        #     return {'beta_a': self.beta_a.detach(), 'beta_b': self.beta_b.detach(), 'b_inf': self.b_inf.detach()}
        # else:

        return {'beta_a': self.beta_a.detach(), 'beta_b': self.beta_b.detach(), 'b_inf': self.b_inf.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"beta_a": self.beta_a.requires_grad,
                            "beta_b": self.beta_b.requires_grad,
                            "b_inf": self.beta_a.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.beta_a.requires_grad_(value["beta_a"])
            self.beta_b.requires_grad_(value["beta_b"])
            self.b_inf.requires_grad_(value["b_inf"])
        else:
            self.beta_a.requires_grad_(value)
            self.beta_b.requires_grad_(value)
            self.b_inf.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.beta_a, self.beta_b, self.b_inf]

    # relevant when beta_a_depth is True
    def get_beta_a_depth_calc(self, **kwargs):
        if self.beta_a_depth:
            return self.beta_a_depth_calc

        else:
            raise ValueError("args.beta_a_depth is False, beta_a_depth Does NOT exist for this case")


@register_operator(name='general_AB')
class GeneralABOperator(LearnableOperator):
    def __init__(self, device, a_mat=1, b_mat=1, a_eta=1, b_eta=1, a_learn_flag=True, b_learn_flag=True,
                 batch_size=1, **kwargs):
        self.device = device
        # self.degmma = kwargs.get("degamma", False)
        self.image_size = kwargs.get("image_size", 256)

        # initialize values for a_mat and b_mat
        if isinstance(a_mat, int) or isinstance(a_mat, float):
            a_mat_tmp = float(a_mat)
        elif isinstance(a_mat, str):
            a_mat_tmp = torch.tensor(np.fromstring(a_mat, dtype=float, sep=','), dtype=torch.float, device=device)
            a_mat_tmp = a_mat_tmp.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError
        self.a_mat = a_mat_tmp * torch.ones(batch_size, 3, self.image_size, self.image_size).to(device)

        if isinstance(b_mat, int) or isinstance(b_mat, float):
            b_mat_tmp = float(b_mat)
        elif isinstance(b_mat, str):
            b_mat_tmp = torch.tensor(np.fromstring(b_mat, dtype=float, sep=','), dtype=torch.float, device=device)
            b_mat_tmp = b_mat_tmp.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError
        self.b_mat = b_mat_tmp * torch.ones(batch_size, 3, self.image_size, self.image_size).to(device)

        self.a_learn_flag = a_learn_flag
        self.b_learn_flag = b_learn_flag

        # coefficients for the Gradient descend step size
        self.a_eta = float(a_eta) if a_learn_flag else float(0)
        self.b_eta = float(b_eta) if b_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.a_mat, "lr": self.a_eta},
                                                                {'params': self.b_mat, "lr": self.b_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 600, 'gamma': 0.1, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)
        # if self.degamma:
        #     rgb_norm = torch.pow(rgb_norm, 2.2)
        # the underwater image formation model
        degraded_image = self.a_mat * rgb_norm + self.b_mat

        return degraded_image, None

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                with torch.no_grad():
                    self.a_mat.add_(self.a_mat.grad, alpha=-self.a_eta)
                    self.b_mat.add_(self.b_mat.grad, alpha=-self.b_eta)

                # zero the gradients so they will not accumulate
                self.a_mat.grad.zero_()
                self.b_mat.grad.zero_()

            else:

                self.optimizer.step()
                self.optimizer.zero_grad()

        # return self.a_mat.detach(), self.b_mat.detach()
        return {'a_mat': self.a_mat.detach(), 'b_mat': self.b_mat.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"a_mat": self.a_mat.requires_grad,
                            "b_mat": self.b_mat.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.a_mat.requires_grad_(value["a_mat"])
            self.b_mat.requires_grad_(value["b_mat"])

        else:
            self.a_mat.requires_grad_(value)
            self.b_mat.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.a_mat, self.b_mat]


@register_operator(name='general_fM')
class GeneralfMOperator(LearnableOperator):
    def __init__(self, device, f_factor=1, m_mat=1, f_eta=1, m_eta=1, f_learn_flag=True, m_learn_flag=True,
                 batch_size=1, variance='original', **kwargs):
        self.device = device
        self.degamma = kwargs.get("degamma", False)
        self.image_size = kwargs.get("image_size", 256)
        self.variance = variance  # original/transmission

        # initialize values for f_factor and m_mat
        if isinstance(f_factor, int) or isinstance(f_factor, float):
            self.f_factor = (float(f_factor) * torch.ones(batch_size, 3, 1, 1)).to(device)
        elif isinstance(f_factor, str):
            self.f_factor = torch.tensor(np.fromstring(f_factor, dtype=float, sep=','), dtype=torch.float,
                                         device=device)
            self.f_factor = self.f_factor.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError

        if isinstance(m_mat, int) or isinstance(m_mat, float):
            m_mat_tmp = float(m_mat)
        elif isinstance(m_mat, str):
            m_mat_tmp = torch.tensor(np.fromstring(m_mat, dtype=float, sep=','), dtype=torch.float, device=device)
            m_mat_tmp = m_mat_tmp.repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError
        self.m_mat = m_mat_tmp * torch.ones(batch_size, 3, self.image_size, self.image_size).to(device)

        self.f_learn_flag = f_learn_flag
        self.m_learn_flag = m_learn_flag

        # coefficients for the Gradient descend step size
        self.f_eta = float(f_eta) if f_learn_flag else float(0)
        self.m_eta = float(m_eta) if m_learn_flag else float(0)

        # set optimizer
        optimizer = kwargs.get("optimizer", None)
        self.optimizer = utilso.get_optimizer(optimizer_name=optimizer,
                                              model_parameters=[{'params': self.f_factor, "lr": self.f_eta},
                                                                {'params': self.m_mat, "lr": self.m_eta}])

        # set scheduler
        scheduler = kwargs.get("scheduler", None)
        scheduler_params = {'step_size': 600, 'gamma': 0.1, 'last_epoch': -1}
        self.scheduler = utilso.get_scheduler(scheduler_name=scheduler, optimizer=self.optimizer, **scheduler_params)

    def forward(self, data, **kwargs):

        # split into rgb and depth
        rgb = data[:, 0:-1, :, :]
        rgb_norm = 0.5 * (rgb + 1)
        if self.degamma:
            rgb_norm = torch.pow(rgb_norm, 2.2)

        if self.variance == "original":
            degraded_image = self.f_factor * rgb_norm + self.m_mat
        elif self.variance == "transmission":
            degraded_image = self.m_mat * rgb_norm + self.f_factor * (1 - self.m_mat)
        else:
            raise ValueError(f"variance is unrecognised: {self.variance}, only 'original' and 'transmission' are valid")

        return degraded_image, None

    def optimize(self, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)

        # when freeze_phi is True that means no optimization is required
        if not freeze_phi:

            # no optimizer was specified - GD is the default
            if self.optimizer is None or self.optimizer == "GD" or self.optimizer == "":

                with torch.no_grad():
                    self.f_factor.add_(self.f_factor.grad, alpha=-self.f_eta)
                    self.m_mat.add_(self.m_mat.grad, alpha=-self.m_eta)

                # zero the gradients so they will not accumulate
                self.f_factor.grad.zero_()
                self.m_mat.grad.zero_()

            else:

                self.optimizer.step()
                self.optimizer.zero_grad()

        # return self.f_factor.detach(), self.m_mat.detach()
        return {'f_factor': self.f_factor.detach(), 'm_mat': self.m_mat.detach()}

    def get_variable_gradients(self, **kwargs):

        grad_enable_dict = {"f_factor": self.f_factor.requires_grad,
                            "m_mat": self.m_mat.requires_grad}

        return grad_enable_dict

    def set_variable_gradients(self, value=None, **kwargs):

        if value is None:
            raise ValueError("A value should be specified (True or False for general or dictionary)")

        if isinstance(value, dict):
            self.f_factor.requires_grad_(value["f_factor"])
            self.m_mat.requires_grad_(value["m_mat"])

        else:
            self.f_factor.requires_grad_(value)
            self.m_mat.requires_grad_(value)

    def get_variable_list(self, **kwargs):

        return [self.f_factor, self.m_mat]


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


@register_noise(name='debka1')
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name='debka2')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


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
