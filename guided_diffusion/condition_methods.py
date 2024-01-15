from abc import ABC, abstractmethod
import torch
import numpy as np
import osmosis_utils.losses as losseso
import osmosis_utils.utils as utilso
import copy

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            difference = measurement - self.operator.forward(x_0_hat[:, 0:3], **kwargs)
            loss = torch.linalg.norm(difference)
            loss_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]
            # loss_grad = torch.autograd.grad(outputs=loss, inputs=x_0_hat)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            loss = torch.linalg.norm(difference) / measurement.abs()
            loss = loss.mean()
            loss_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return loss_grad, loss

    @abstractmethod
    # def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        pass


@register_conditioning_method(name='osmosis')
class PosteriorSamplingOsmosis(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        input_scale_str = kwargs.get('scale', 1.0)

        # in case scale is single for all channels
        try:
            self.scale = torch.tensor([float(input_scale_str)])

        # in case there is a scale value for each channel
        except ValueError:
            self.scale = torch.tensor([float(num_str.strip()) for num_str in input_scale_str.split(',')])

        self.gradient_x_prev = kwargs.get('gradient_x_prev', False)

        # sample pattern parameters
        self.pattern_name = kwargs.get('pattern', 'original')
        self.global_N = kwargs.get('global_N', 1)
        self.local_M = kwargs.get('local_M', 1)
        self.n_iter = kwargs.get('n_iter', 1)
        self.update_start = kwargs.get('update_start', 1.0)

        # Auxiliary loss information
        aux_loss_dict = kwargs.get("aux_loss", None)
        if aux_loss_dict is not None:
            aux_loss_dict = {key_ii: float(value_ii) for key_ii, value_ii in aux_loss_dict.items()}
            # Quality loss object
            self.aux_loss = losseso.AuxiliaryLoss(aux_loss_dict)
        else:
            self.aux_loss = None

        # guiding loss function, loss weight (depth or none), is depth - what function and values
        self.loss_function = kwargs.get("loss_function", "norm")
        self.loss_weight = kwargs.get("loss_weight", None)
        self.weight_function = kwargs.get("weight_function", None)

        # use gradient clipping (or image clipping)
        gradient_clip_tmp = kwargs.get("gradient_clip", "False")
        gradient_clip_tmp = [num_str for num_str in gradient_clip_tmp.split(',')]
        self.gradient_clip = utilso.str2bool(gradient_clip_tmp[0])

        # if true - what values
        if self.gradient_clip:
            self.gradient_clip_value = float(gradient_clip_tmp[1].strip())
        else:
            self.gradient_clip_value = None

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):

        # compute the degraded image on the unet prediction (operator) - in measurement file
        degraded_image_tmp = self.operator.forward(x_0_hat, **kwargs)

        # back to [-1,1]
        degraded_image = 2 * degraded_image_tmp - 1

        differance = (measurement - degraded_image)

        # create the loss weights - multiply the differences
        loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight,
                                             weight_function=self.weight_function,
                                             degraded_image=degraded_image_tmp.detach(),
                                             x_0_hat=x_0_hat.detach())
        differance = differance * loss_weight

        # loss function - norm2
        if self.loss_function == 'norm':
            loss = torch.linalg.norm(differance)
            # calculated for visualization
            sep_loss = torch.norm(differance.detach().cpu(), p=2, dim=[1, 2, 3]).numpy()

        # Mean square error
        elif self.loss_function == "mse":
            mse = differance ** 2
            mse = mse.mean(dim=(1, 2, 3))
            loss = mse.sum()
            # calculated for visualization
            sep_loss = mse.detach().cpu().numpy()

        # No other loss
        else:
            raise NotImplementedError

        return sep_loss, loss, degraded_image_tmp.detach()

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)
        time_index = kwargs.get("time_index", None)

        # when the gradient is w.r.t x0, the x_prev gradients and history of the x0 prediction are not required
        if not self.gradient_x_prev:
            x_0_hat = x_0_hat.detach().to(x_0_hat.device)
            x_0_hat.requires_grad_(True)
            x_0_hat = x_0_hat.to(x_0_hat.device)
            x_prev.requires_grad_(False)

        # calculate the losses
        with torch.set_grad_enabled(True):

            # phi's require gradients when we update them, hence when freeze_phi is False
            self.operator.set_variable_gradients(value=not freeze_phi)

            # the number of inner optimization num of steps should be 1 if freezing phi,
            # since there is no optimizing at all in this case
            inner_optimize_length = 1 if freeze_phi else self.n_iter

            for optimize_ii in range(inner_optimize_length):

                # compute the loss after applying the operator, sep_loss is relevant for multiple images
                sep_loss, loss, degraded_image_01 = self.grad_and_value(x_prev=x_prev,
                                                                        x_0_hat=x_0_hat,
                                                                        measurement=measurement,
                                                                        time_index=time_index)

                # total loss refers to the original loss or to the loss of the x
                if self.aux_loss is not None:
                    aux_loss, aux_loss_dict = self.aux_loss.forward(x_0_hat)
                    total_loss = loss + aux_loss
                else:
                    aux_loss_dict = None
                    total_loss = loss

                # calculate the backward graph
                if optimize_ii == (inner_optimize_length - 1):
                    if freeze_phi:
                        # calculate graph w.r.t x_prev
                        total_loss.backward(inputs=[x_prev])
                    else:
                        # calculate graph w.r.t x_prev and phi's
                        total_loss.backward(inputs=[x_prev] + self.operator.get_variable_list())
                else:
                    # when optimize only the phi's, we specify it for faster run time
                    total_loss.backward(inputs=self.operator.get_variable_list())

                # optimize phi's, in case of freeze phi true - optimization is not done
                variables_dict = self.operator.optimize(freeze_phi=freeze_phi)

            # update x_t
            with torch.no_grad():

                # remove - opher
                # # update guidance scale
                # scale_norm = utilso.set_guidance_scale_norm(norm_type=self.scale_norm, x_0_hat=x_0_hat.detach(),
                #                                             x_t=x_t.detach(), x_prev=x_prev,
                #                                             sample_added_noise=sample_added_noise)
                # # reshape the scale according to [b,c,h,w]
                # guidance_scale = scale_norm * self.scale[None, ..., None, None].to(x_prev.device)

                # reshape the scale according to [b,c,h,w]
                guidance_scale = self.scale[None, ..., None, None].to(x_prev.device)

                # update x_t - gradient w.r.t x_t
                if self.gradient_x_prev:

                    if self.gradient_clip:
                        grads = torch.clamp(x_prev.grad,
                                            min=-self.gradient_clip_value,
                                            max=self.gradient_clip_value)
                    else:
                        grads = x_prev.grad

                    x_t -= guidance_scale * grads
                    gradients = x_prev.grad.cpu()

                # update x_t - gradient w.r.t x_0_pred
                else:
                    x_t -= guidance_scale * x_0_hat.grad
                    gradients = x_0_hat.grad.cpu()

        return x_t, sep_loss, variables_dict, gradients, aux_loss_dict


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)

        input_scale_str = kwargs.get('scale', 1.0)
        # in case scale is single for all channels
        try:
            self.scale = torch.tensor([float(input_scale_str)])
        # in case there is a scale value for each channel
        except ValueError:
            self.scale = torch.tensor([float(num_str.strip()) for num_str in input_scale_str.split(',')])

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale[None, ..., None, None].to(x_prev.device)

        return x_t, norm
