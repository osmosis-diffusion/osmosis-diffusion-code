import os
import numpy as np
import torch
import torch.nn as nn

# %% base functions for getting loss

__LOSS__ = {}


def register_loss(name: str):
    def wrapper(cls):
        if __LOSS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __LOSS__[name] = cls
        return cls

    return wrapper


def get_loss(name: str, **kwargs):
    if __LOSS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __LOSS__[name](**kwargs)


# %% global exposure loss

@register_loss(name='avrg_loss')
class Average_Loss(nn.Module):
    """
    Global Exposure Control Loss
    """

    def __init__(self):
        super(Average_Loss, self).__init__()

    def forward(self, x):
        #  only color data (rgb) is required, depth is not required here - value should be [-1,1]

        x_norm = x[:, 0:3, :, :]
        mean = torch.mean(x_norm, dim=(2, 3))
        avrg_loss = torch.sum(torch.abs(mean))

        return avrg_loss


# %% Value loss

@register_loss(name='val_loss')
class Value_Loss(nn.Module):

    def __init__(self, device=torch.device("cuda:0"), **kwargs):
        super(Value_Loss, self).__init__()
        self.device = torch.device(device)

    def forward(self, rgbd, **kwargs):
        rgb = (rgbd[:, 0:3, :, :])
        value = kwargs.get("value", 0.7)
        val_loss = (torch.maximum(rgb.abs() - value, torch.zeros_like(rgb)) ** 2).mean()

        return val_loss


# %% Auxiliary loss class which includes all the quality losses and their coefficients

class AuxiliaryLoss(nn.Module):
    def __init__(self, losses_dictionary):
        super(AuxiliaryLoss, self).__init__()

        self.losses_dictionary = losses_dictionary
        self.losses_list = [get_loss(key_ii) for key_ii in losses_dictionary.keys()]
        self.loss_gammas = [torch.tensor(value_ii) for value_ii in losses_dictionary.values()]

    def forward(self, x):
        aux_loss = 0
        aux_loss_dict = {}
        # summing the losses according to their gammas
        for gamma_ii, loss_ii, loss_name_ii in zip(self.loss_gammas, self.losses_list, self.losses_dictionary):
            cur_loss = loss_ii.forward(x)
            aux_loss += gamma_ii.to(x.device) * cur_loss
            aux_loss_dict[loss_name_ii] = cur_loss.detach().cpu()
        return aux_loss, aux_loss_dict
