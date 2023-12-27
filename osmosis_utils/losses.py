import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.vgg import vgg16

import osmosis_utils.utils as utilso
import osmosis_utils.histograms as histo

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


# %% losses from GDP paper - I did not use them yet

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_TV_bak(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        print(x.size)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = 1

        h_x = x.size()[0]
        w_x = x.size()[1]

        count_h = (x.size()[0] - 1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)

        h_tv = torch.pow((x[1:, :] - x[:h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, 1:] - x[:, :w_x - 1]), 2).sum()

        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# %% color constancy loss
@register_loss(name='color')
class L_color(nn.Module):
    """
    Color Constancy Loss from GDP paper (section 5 - Quality Enhancement Loss)
    """

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        # normalizing to 0,1 ; only color data (rgb) is required, depth is not required here
        x_norm = 0.5 * (x[:, 0:3, :, :] + 1)

        # mean value for each color channel
        mean_rgb = torch.mean(x_norm, dim=[2, 3])
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)

        # loss_color = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)[0, 0]
        loss_color = (Drg + Drb + Dgb)[0, 0]

        return loss_color


# %% color constancy loss with depth weights

@register_loss(name='color_depth')
class L_color_depth(nn.Module):
    """
    Color Constancy Loss from GDP paper (section 5 - Quality Enhancement Loss)
    but scaled with the depths value
    """

    def __init__(self):
        super(L_color_depth, self).__init__()

    def forward(self, x):
        # normalizing to 0,1 ; only color data (rgb) is required, depth is not required here
        x_norm = 0.5 * (x[:, 0:3, :, :] + 1)

        # depth normalized
        x_depth = x[:, 3, :, :].detach().unsqueeze(1)
        # x_depth = x_depth + 1.5
        # x_depth = (x_depth + 1.4) * 1.4
        x_depth_norm = torch.clamp(x_depth + 1.7, min=0.5, max=2.5)

        # mean value for each color channel
        mean_rgb = torch.mean(x_norm * x_depth_norm, dim=[2, 3])
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)

        # loss_color = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)[0, 0]
        loss_color = Drg + Drb + Dgb

        return loss_color


# %% local exposure loss

@register_loss(name='exposure')
class L_exp(nn.Module):
    """
    Exposure Control Loss from GDP paper (section 5 - Quality Enhancement Loss)
    """

    def __init__(self, patch_size=8, mean_val=0.2):
        super(L_exp, self).__init__()

        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = torch.FloatTensor([mean_val])

    def forward(self, x):
        # normalizing to 0,1 ; only color data (rgb) is required, depth is not required here
        x_norm = 0.5 * (x[:, 0:3, :, :] + 1)

        x_norm = torch.mean(x_norm, dim=1)
        mean = self.pool(x_norm)

        # loss_exp = torch.mean(torch.pow(mean - self.mean_val.to(x.device), 2))
        loss_exp = torch.mean(torch.abs(mean - self.mean_val.to(x.device)))

        return loss_exp


# %% global exposure loss

@register_loss(name='exposure_global')
class L_exp_global(nn.Module):
    """
    Global Exposure Control Loss - not per patch
    """

    def __init__(self):
        super(L_exp_global, self).__init__()

    def forward(self, x):
        #  only color data (rgb) is required, depth is not required here - value should be [-1,1]
        x_norm = x[:, 0:3, :, :]

        mean = torch.mean(x_norm, dim=(2, 3))
        # loss_exp = torch.mean(torch.pow(mean - self.mean_val.to(x.device), 2))
        loss_exp = torch.sum(torch.abs(mean))

        return loss_exp


# %% perception loss
@register_loss(name='perception')
class Perception_Loss(nn.Module):
    def __init__(self):
        super(Perception_Loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


# %% red channel prior loss
@register_loss(name='red_channel_prior')
class Red_Channel_Prior_Loss(nn.Module):

    def __init__(self):
        super(Red_Channel_Prior_Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, mask):

        rcp = x[:, 1:3, :, :].max(dim=1) - x[:, 0, :, :]

        if len(mask.shape) == 2:
            return self.loss(rcp, mask)
        elif mask.shape[-3] == 3:
            return self.loss(rcp, mask.mean(dim=0))


# %% histogram loss

@register_loss(name='histogram')
class Histogram_Loss(nn.Module):

    def __init__(self,
                 histogram_target_path=r"D:\datasets\images_samples\histograms\imagenet_sub_mean\imagenet_sub_mean.npy",
                 device=torch.device("cuda")):
        super(Histogram_Loss, self).__init__()
        # r"D:\datasets\images_samples\histograms\imagenet_sub_mean\imagenet_sub_mean.npy"
        self.histogram_target_path = histogram_target_path
        self.histogram_target = torch.tensor(np.load(histogram_target_path)).to(device)

        # histograms parameters
        self.intensity_scale = True
        self.histogram_size = 64
        self.max_input_size = 256
        self.method = 'inverse-quadratic'

        self.loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor
        self.unloader = transforms.ToPILImage()  # reconvert into PIL image

        # histogram object
        self.histogram_block = histo.RGBuvHistBlock(insz=self.max_input_size, h=self.histogram_size,
                                                    intensity_scale=self.intensity_scale,
                                                    method=self.method,
                                                    device=device)

    def forward(self, rgbd):
        rgb = 0.5 * (rgbd[:, 0:3, :, :] + 1)
        hist_image = self.histogram_block(rgb)
        histogram_loss = histo.histogram_loss(hist_image, target_hist=self.histogram_target)
        return histogram_loss


# %% histogram loss - equalization

@register_loss(name='histogram_equal')
class Histogram_Equalization_Loss(nn.Module):

    def __init__(self, device=torch.device("cuda")):
        super(Histogram_Equalization_Loss, self).__init__()
        self.device = torch.device(device)
        self.N = 256

    def forward(self, rgbd):
        v = torch.arange(self.N, device=self.device)
        rgb = 255 * 0.5 * (rgbd[:, 0:3, :, :] + 1)
        h = {}

        for i in range(3):
            h[i] = torch.mean(0.5 * torch.exp(-0.5 * (v[None] - rgb[:, i, :, :].reshape(-1)[:, None]) ** 2), dim=0)

        histogram_loss = torch.sum((h[0] - 1 / self.N) ** 2 + (h[1] - 1 / self.N) ** 2 + (h[2] - 1 / self.N) ** 2)

        return histogram_loss


# %% Simon loss

@register_loss(name='simon_loss')
class Simon_Loss(nn.Module):

    def __init__(self, device=torch.device("cuda:0"), **kwargs):
        super(Simon_Loss, self).__init__()
        self.device = torch.device(device)

    def forward(self, rgbd, **kwargs):
        rgb = (rgbd[:, 0:3, :, :])
        # value = kwargs.get("value", 0.6)

        # in 15-11-23 run19, run20 it was 0.7
        value = kwargs.get("value", 0.7)

        simon_loss = (torch.maximum(rgb.abs() - value, torch.zeros_like(rgb)) ** 2).mean()

        return simon_loss


# %% Quality loss class which includes all the quality losses and their coefficients

class QualityLoss(nn.Module):
    def __init__(self, losses_dictionary):
        super(QualityLoss, self).__init__()

        self.losses_dictionary = losses_dictionary
        self.losses_list = [get_loss(key_ii) for key_ii in losses_dictionary.keys()]
        self.loss_gammas = [torch.tensor(value_ii) for value_ii in losses_dictionary.values()]

    def forward(self, x):
        quality_loss = 0
        quality_loss_dict = {}
        # summing the losses according to their gammas
        for gamma_ii, loss_ii, loss_name_ii in zip(self.loss_gammas, self.losses_list, self.losses_dictionary):
            cur_loss = loss_ii.forward(x)
            quality_loss += gamma_ii.to(x.device) * cur_loss
            quality_loss_dict[loss_name_ii] = cur_loss.detach().cpu()
        return quality_loss, quality_loss_dict
