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


@register_conditioning_method(name='gdp')
class PosteriorSamplingGDP(ConditioningMethod):
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

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            # original
            # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # norm = torch.linalg.norm(difference)
            # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

            # debka version
            x_0_hat_norm = 0.5 * (x_0_hat + 1)
            degraded_image = self.operator.forward(x_0_hat_norm, **kwargs)
            degraded_image = 2 * degraded_image - 1

            loss = torch.linalg.norm(measurement - degraded_image)

            if self.gradient_x_prev:
                # calculate gradient w.r.t x_t (like GDP_xt and DPS)
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_prev, retain_graph=True)[0]
            else:
                # calculate gradient w.r.t x_0_hat (like GDP_x0)
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_0_hat, retain_graph=True)[0]

        elif self.noiser.__name__ == 'poisson':

            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif (self.noiser.__name__ == 'debka1') or (self.noiser.__name__ == 'debka2'):

            # set the images to be in range of [0,1] instead of [-1,1], the reason is the values of b_inf and beta
            # TODO: I am not sure that this is a correct thing to do, since the loss that calculated will use to
            #  optimize the sampling witch was done for an image in the range [-1,1]

            x_0_hat_norm = 0.5 * (x_0_hat + 1)
            # measurement = 0.5 * (measurement + 1)

            degraded_image = self.operator.forward(x_0_hat_norm, **kwargs)
            degraded_image = 2 * degraded_image - 1
            # loss = torch.nn.functional.mse_loss(measurement, degraded_image)
            mse = (degraded_image - measurement) ** 2
            mse = mse.mean(dim=(1, 2, 3))
            loss = mse.sum()

            if self.gradient_x_prev:
                # calculate gradient w.r.t x_t (like GDP_xt and DPS)
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_prev, retain_graph=True)[0]
            else:
                # calculate gradient w.r.t x_0_hat (like GDP_x0)
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_0_hat, retain_graph=True)[0]

            # TODO:
            #   2. if the update is here, it is like GDP - I choose DPS way
            #   3. if the update is like DPS, we have to put it outside - I choose this!
            #   4. check out GDP code - weird things happens there and we have to check it -
            #       I sent question in the github rep

            # optimize beta and b_inf - as GDP
            # self.operator.optimize()

        else:
            raise NotImplementedError

        return loss_grad, loss

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):

        with torch.enable_grad():
            loss_grad, loss = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)

            # this is "moving" the sample according to the sampling - it happens after the sampling (DPS)
            # in GDP it happens before sampling

            # et ze dborah limda oty - hi Hachama beramot
            x_t -= self.scale[None, ..., None, None].to(loss_grad.device) * loss_grad

            # TODO:
            #   4. if the update is like DPS, we have to put it here
            #       it is possible because the operator is inside the self.operator

            # optimize beta and b_inf - as DPS
            beta, b_inf = self.operator.optimize(loss)

        return x_t, loss, beta, b_inf


@register_conditioning_method(name='gdp_pattern')
class PosteriorSamplingGDPpattern(ConditioningMethod):
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

        self.backward_guidance = kwargs.get('backward_guidance', None)

        self.loss_like_dps_paper = kwargs.get('loss_like_dps_paper', False)

        self.scale_norm = kwargs.get('scale_norm', None)
        self.mask_depth = kwargs.get('mask_depth', False)
        # Quality loss information
        quality_loss_dict = kwargs.get("quality_loss", None)
        if quality_loss_dict is not None:
            quality_loss_dict = {key_ii: float(value_ii) for key_ii, value_ii in quality_loss_dict.items()}
            # Quality loss object
            self.quality_loss = losseso.QualityLoss(quality_loss_dict)
        else:
            self.quality_loss = None

        self.loss_function = kwargs.get("loss_function", "norm")
        self.loss_weight = kwargs.get("loss_weight", None)
        self.weight_function = kwargs.get("weight_function", None)

        gradient_clip_tmp = kwargs.get("gradient_clip", "False")
        gradient_clip_tmp = [num_str for num_str in gradient_clip_tmp.split(',')]
        self.gradient_clip = utilso.str2bool(gradient_clip_tmp[0])
        if self.gradient_clip:
            self.gradient_clip_values = [float(gradient_clip_tmp[1].strip()), float(gradient_clip_tmp[2].strip())]
        else:
            self.gradient_clip_values = None

        change_clipping_tmp = kwargs.get("change_clipping", 0)
        self.change_clipping = change_clipping_tmp if self.gradient_clip else None
        self.simon_and_clip = kwargs.get("simon_and_clip", False)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):

        backward_guidance = kwargs.get("backward_guidance", None)
        time_index = kwargs.get("time_index", None)

        # in case we want backward guidance (universal guidance paper)
        if (backward_guidance is not None) and (backward_guidance["backward_guidance"]) and (
                backward_guidance["back_start"] >= time_index >= backward_guidance["back_end"]):

            x_0_hat_op = x_0_hat.detach()
            x_0_hat_op.requires_grad_(True)
            x_0_hat_op = x_0_hat_op.to(x_0_hat.device)

            # remember the previous "requires_grad" states of the other variables
            previous_variables_gradient = self.operator.get_variable_gradients()
            x_prev_req_grad = x_prev.requires_grad

            # apply "requires_grad" on x_0_hat only
            self.operator.set_variable_gradients(value=False)
            x_prev.requires_grad_(False)

            # define optimizer and scheduler
            x_0_optimizer = torch.optim.Adam([x_0_hat_op], lr=float(backward_guidance["lr"]))
            x_0_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=x_0_optimizer,
                                                                       T_max=backward_guidance['num_iters'])

            # loop over the number of iterations
            for back_guide_ii in range(backward_guidance['num_iters']):
                x_0_optimizer.zero_grad()
                # compute the degraded image on the unet prediction (operator)
                degraded_image_tmp, mask_large_depth = self.operator.forward(x_0_hat_op, **kwargs)
                # back to [-1,1]
                degraded_image = 2 * degraded_image_tmp - 1
                # calculate loss
                loss = torch.linalg.norm(degraded_image - measurement)
                # calculate backward graph - gradient
                loss.backward(inputs=[x_0_hat_op])
                # update x_0_hat
                x_0_optimizer.step()
                # scheduler step
                x_0_scheduler.step()

            # update "requires_grad" states of the other variables for the general loss calculation
            self.operator.set_variable_gradients(value=previous_variables_gradient)
            x_prev.requires_grad_(x_prev_req_grad)
            x_0_hat.data = x_0_hat_op.detach()
            x_0_hat.requires_grad_(x_prev_req_grad)

        # compute the degraded image on the unet prediction (operator)
        degraded_image_tmp, mask_large_depth = self.operator.forward(x_0_hat, **kwargs)

        # back to [-1,1]
        degraded_image = 2 * degraded_image_tmp - 1

        # masking the differences according to too large depth values
        differance = (measurement - degraded_image)
        if self.mask_depth:
            differance *= mask_large_depth

        # create the loss weights - multiply th differences
        loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight,
                                             weight_function=self.weight_function,
                                             degraded_image=degraded_image_tmp.detach(),
                                             x_0_hat=x_0_hat.detach())
        # else:
        differance = differance * loss_weight

        # loss function
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
        sample_added_noise = kwargs.get("sample_added_noise", None)
        time_index = kwargs.get("time_index", None)

        if self.gradient_clip and time_index < self.change_clipping:
            x_0_hat = torch.clamp(x_0_hat, -1., 1.)

        # when the gradient is w.r.t x0 there is no meed of the x_prev gradients and history of the x0 prediction
        if not self.gradient_x_prev:
            x_0_hat = x_0_hat.detach().to(x_0_hat.device)
            x_0_hat.requires_grad_(True)
            x_0_hat = x_0_hat.to(x_0_hat.device)
            x_prev.requires_grad_(False)

        # calculate the losses
        with torch.set_grad_enabled(True):

            # betas and b_inf are required gradients when we update them, hence when freeze_phi is False
            self.operator.set_variable_gradients(value=not freeze_phi)

            # the number of inner optimization num of steps should be 1 if freezing phi,
            # since there is no optimizing at all in this case
            inner_optimize_length = 1 if freeze_phi else self.n_iter

            for optimize_ii in range(inner_optimize_length):

                # compute the loss after applying the operator, sep_loss is relevant for multiple images
                sep_loss, loss, degraded_image_01 = self.grad_and_value(x_prev=x_prev,
                                                                        x_0_hat=x_0_hat,
                                                                        measurement=measurement,
                                                                        backward_guidance=self.backward_guidance,
                                                                        time_index=time_index)

                # total loss refers to the original loss or to the loss of the x
                if self.quality_loss is not None:
                    quality_loss, quality_loss_dict = self.quality_loss.forward(x_0_hat)
                    total_loss = loss + quality_loss
                else:
                    quality_loss_dict = None
                    total_loss = loss

                # calculate the backward graph
                if optimize_ii == (inner_optimize_length - 1):
                    if freeze_phi:
                        total_loss.backward(inputs=[x_prev])
                    else:
                        total_loss.backward(inputs=[x_prev] + self.operator.get_variable_list())
                else:
                    # when optimize only the betas and b_inf, we specify it for faster run time
                    # total_loss.backward(inputs=[self.operator.beta_a, self.operator.beta_b, self.operator.b_inf])
                    total_loss.backward(inputs=self.operator.get_variable_list())

                # optimize the betas, b_inf, in case of freeze phi, optimization is not done
                # beta, b_inf = self.operator.optimize(freeze_phi=freeze_phi)
                variables_dict = self.operator.optimize(freeze_phi=freeze_phi)

            # step the scheduler
            if self.operator.scheduler is not None:
                self.operator.scheduler.step()

            # update x_t
            with torch.no_grad():

                # osmosis_utils is checking something
                # x_t = x_t + sample_added_noise

                # update guidance scale
                scale_norm = utilso.set_guidance_scale_norm(norm_type=self.scale_norm,
                                                            x_0_hat=x_0_hat.detach(), x_t=x_t.detach(),
                                                            x_prev=x_prev,
                                                            sample_added_noise=sample_added_noise)
                # print(f"scale_norm: {scale_norm.squeeze().cpu()}")
                guidance_scale = scale_norm * self.scale[None, ..., None, None].to(x_prev.device)

                # update x_t
                if self.gradient_x_prev:
                    # grads_norm_tmp = torch.linalg.norm(x_prev.grad[:, 0:3, :, :], dim=1).to(torch.device("cpu"))
                    # print(f"\nBEFORE:"
                    #       f"\nmin: {grads_norm_tmp.min()}, max: {grads_norm_tmp.max()},"
                    #       f"mean: {grads_norm_tmp.mean()} ,std: {grads_norm_tmp.std()}"
                    #       f"\nRGB: min: {x_prev.grad[:, 0:3, :, :].min()}, max: {x_prev.grad[:, 0:3, :, :].max()},"
                    #       f"mean: {x_prev.grad[:, 0:3, :, :].mean()} ,std: {x_prev.grad[:, 0:3, :, :].std()}"
                    #       f"\nDepth: min: {x_prev.grad[:, 3, :, :].min()}, max: {x_prev.grad[:, 3, :, :].max()},"
                    #       f"mean: {x_prev.grad[:, 3, :, :].mean()} ,std: {x_prev.grad[:, 3, :, :].std()}")

                    if self.gradient_clip and time_index >= self.change_clipping:

                        # grads = torch.zeros_like(x_prev.grad, device=x_prev.grad.device)
                        # grads[:, 0:3, :, :] = utilso.gradient_clip_norm(x_prev.grad[:, 0:3, :, :],
                        #                                                 max_value=self.gradient_clip_values[0])
                        # grads[:, 3, :, :] = torch.clamp(x_prev.grad[:, 3, :, :], min=-self.gradient_clip_values[1],
                        #                                 max=self.gradient_clip_values[1])
                        #

                        grads = torch.clamp(x_prev.grad, min=-self.gradient_clip_values[1],
                                            max=self.gradient_clip_values[1])

                        # grads_new_norm = torch.linalg.norm(grads[:, 0:3, :, :], dim=1).to(torch.device("cpu"))

                        # print(f"\nAFTER:"
                        #       f"\nmin: {grads_new_norm.min()}, max: {grads_new_norm.max()},"
                        #       f"mean: {grads_new_norm.mean()} ,std: {grads_new_norm.std()}"
                        #       f"\nRGB: min: {grads[:, 0:3, :, :].min()}, max: {grads[:, 0:3, :, :].max()},"
                        #       f"mean: {grads[:, 0:3, :, :].mean()} ,std: {grads[:, 0:3, :, :].std()}"
                        #       f"\nDepth: min: {grads[:, 3, :, :].min()}, max: {grads[:, 3, :, :].max()},"
                        #       f"mean: {grads[:, 3, :, :].mean()} ,std: {grads[:, 3, :, :].std()}")

                    else:
                        grads = x_prev.grad

                        # x_t -= guidance_scale * x_prev.grad
                    x_t -= guidance_scale * grads
                    gradients = x_prev.grad.cpu()
                else:
                    x_t -= guidance_scale * x_0_hat.grad
                    gradients = x_0_hat.grad.cpu()

            # new grad - zero the gradients after update zero the relevant gradients - I don't think this is required
            # _ = x_prev.grad.zero_() if self.gradient_x_prev else x_0_hat.grad.zero_()

        # return x_t, sep_loss, beta, b_inf, gradients
        return x_t, sep_loss, variables_dict, gradients, quality_loss_dict


@register_conditioning_method(name='gdp_pattern_diff_loss')
class PosteriorSamplingGDPpattern_diff_loss(ConditioningMethod):
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

        self.backward_guidance = kwargs.get('backward_guidance', None)

        self.loss_like_dps_paper = kwargs.get('loss_like_dps_paper', False)

        self.scale_norm = kwargs.get('scale_norm', None)
        self.mask_depth = kwargs.get('mask_depth', False)
        # Quality loss information
        quality_loss_dict = kwargs.get("quality_loss", None)
        if quality_loss_dict is not None:
            quality_loss_dict = {key_ii: float(value_ii) for key_ii, value_ii in quality_loss_dict.items()}
            # Quality loss object
            self.quality_loss = losseso.QualityLoss(quality_loss_dict)
        else:
            self.quality_loss = None

        self.loss_function = kwargs.get("loss_function", "norm")

        # loss weight for the original case and *NOT* for the "phi"
        self.loss_weight = kwargs.get("loss_weight", None)
        self.weight_function = kwargs.get("weight_function", None)
        # loss weight for the "phi"s
        self.loss_weight_diff = kwargs.get("loss_weight_diff", None)
        self.weight_function_diff = kwargs.get("weight_function_diff", None)

        # diff_vars_tmp = kwargs.get("diff_vars", ['beta_a', 'beta_b', 'b_inf'])
        diff_vars_tmp = kwargs.get("diff_vars", [])
        self.diff_vars = diff_vars_tmp.split(",") if isinstance(diff_vars_tmp, str) else diff_vars_tmp

        # update only the variables in phi list:
        self.diff_vars_list, self.non_diff_vars_list = [], []
        if "beta_a" in self.diff_vars:
            self.diff_vars_list.append(self.operator.beta_a)
        else:
            self.non_diff_vars_list.append(self.operator.beta_a)
        if "beta_b" in self.diff_vars:
            self.diff_vars_list.append(self.operator.beta_b)
        else:
            self.non_diff_vars_list.append(self.operator.beta_b)
        if "b_inf" in self.diff_vars:
            self.diff_vars_list.append(self.operator.b_inf)
        else:
            self.non_diff_vars_list.append(self.operator.b_inf)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):

        # compute the degraded image on the unet prediction (operator)
        degraded_image_tmp, mask_large_depth = self.operator.forward(x_0_hat, **kwargs)

        # back to [-1,1]
        degraded_image = 2 * degraded_image_tmp - 1

        # masking the differences according to too large depth values
        differance = (measurement - degraded_image)
        if self.mask_depth:
            differance *= mask_large_depth

        # create the loss weights - multiply th differences
        loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight,
                                             degraded_image=degraded_image_tmp.detach(),
                                             x_0_hat=x_0_hat.detach())
        # if self.different_loss_phi:
        #     differance_phi = differance.clone()
        #     differance = loss_weight * differance
        #
        #     loss = []
        #     # loss function
        #     if self.loss_function == 'norm':
        #         loss.append(torch.linalg.norm(differance))
        #         # calculated for visualization
        #         # sep_loss = torch.norm(differance.detach().cpu(), p=2, dim=[1, 2, 3])
        #
        #         loss.append(torch.linalg.norm(differance_phi))
        #
        #         sep_loss = np.array([loss[0].detach().cpu().numpy(), loss[1].detach().cpu().numpy()])
        #
        #     # Mean square error
        #     elif self.loss_function == "mse":
        #
        #         mse = differance ** 2
        #         mse = mse.mean(dim=(1, 2, 3))
        #         loss.append(mse.sum())
        #         # calculated for visualization
        #         sep_loss = mse.detach().cpu()
        #
        #         mse = differance_phi ** 2
        #         mse = mse.mean(dim=(1, 2, 3))
        #         loss.append(mse.sum())
        #
        #     # No other loss
        #     else:
        #         raise NotImplementedError

        # else:

        differance = differance * loss_weight

        # loss function
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

    def forward_and_diff(self, x_0_hat, measurement):

        # compute the degraded image on the unet prediction (operator)
        degraded_image_tmp, mask = self.operator.forward(x_0_hat)
        # back to [-1,1]
        degraded_image = 2 * degraded_image_tmp - 1
        # masking the differences according to too large depth values
        differance = (measurement - degraded_image)

        return differance, mask, degraded_image_tmp

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):

        freeze_phi = kwargs.get("freeze_phi", False)
        sample_added_noise = kwargs.get("sample_added_noise", None)
        sep_loss = []

        # when the gradient is w.r.t x0 there is no need of the x_prev gradients and history of the x0 prediction
        if not self.gradient_x_prev:
            x_0_hat = x_0_hat.detach().requires_grad_(True).to(x_0_hat.device)
            x_prev.requires_grad_(False)
            # self.non_phi_list.append(x_0_hat)
        # else:
        # self.non_phi_list.append(x_prev)

        # calculate the losses
        with torch.set_grad_enabled(True):

            # betas and b_inf are required gradients when we update them, hence when freeze_phi is False
            self.operator.set_variable_gradients(value=not freeze_phi)

            # the number of inner optimization num of steps should be 1 if freezing phi,
            # since there is no optimizing at all in this case
            inner_optimize_length = 1 if freeze_phi else self.n_iter

            for optimize_ii in range(inner_optimize_length):

                if not freeze_phi:
                    # first calculate loss for the ** diff vars **
                    differance, mask, uw_image = self.forward_and_diff(x_0_hat, measurement)
                    # create the loss weights - multiply th differences
                    loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight_diff,
                                                         weight_function=self.weight_function_diff,
                                                         degraded_image=uw_image.detach(),
                                                         x_0_hat=x_0_hat.detach())
                    # loss function
                    loss = utilso.compute_loss(self.loss_function, differance, loss_weight)
                    # calculate graph
                    loss.backward(inputs=self.diff_vars_list)

                if optimize_ii == (inner_optimize_length - 1):
                    x_prev.requires_grad_(True)
                    update_list = [x_prev] if freeze_phi else self.non_diff_vars_list + [x_prev]
                else:
                    x_prev.requires_grad_(False)
                    update_list = self.non_diff_vars_list

                # second calculate loss for the ** non diff vars **
                differance, mask, uw_image = self.forward_and_diff(x_0_hat, measurement)
                # create the loss weights - multiply th differences
                loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight,
                                                     weight_function=self.weight_function,
                                                     degraded_image=uw_image.detach(),
                                                     x_0_hat=x_0_hat.detach())
                # loss function
                loss = utilso.compute_loss(self.loss_function, differance, loss_weight)
                sep_loss = [loss.detach().cpu()]
                # calculate backward graph
                loss.backward(inputs=update_list)

                # loss.backward(inputs=self.non_diff_vars_list)

                # optimize the betas, b_inf, in case of freeze phi, optimization is not done
                variables_dict = self.operator.optimize(freeze_phi=freeze_phi)

            # step the scheduler
            if (self.operator.scheduler is not None) and (not freeze_phi):
                self.operator.scheduler.step()

            # if freeze_phi:
            #     # enable grad of x_prev
            #     x_prev.requires_grad_(True)
            #     # disable grad of phi
            #     self.operator.beta_a.requires_grad_(False)
            #     self.operator.beta_b.requires_grad_(False)
            #     self.operator.b_inf.requires_grad_(False)
            #
            # else:
            #     # disable grad of non_phi_list and enable grad of phi_list
            #     _ = [ii.requires_grad_(True) for ii in self.non_phi_list]
            #     _ = [ii.requires_grad_(False) for ii in self.phi_list]

            # # set gradients
            # self.operator.set_variable_gradients(value=False)
            # x_prev.requires_grad_(freeze_phi)
            # # compute loss for updating x_t
            # differance, mask, uw_image = self.forward_and_diff(x_0_hat, measurement)
            # # create the loss weights - multiply th differences
            # loss_weight = utilso.set_loss_weight(loss_weight_type=self.loss_weight,
            #                                      weight_function=self.weight_function,
            #                                      degraded_image=uw_image.detach(),
            #                                      x_0_hat=x_0_hat.detach())
            # # loss function
            # loss = utilso.compute_loss(self.loss_function, differance, loss_weight)
            # sep_loss = loss.detach().cpu()
            # if self.quality_loss is not None:
            #     quality_loss, quality_loss_dict = self.quality_loss.forward(x_0_hat)
            #     loss = loss + quality_loss
            #
            # # calculate graph
            # loss.backward(inputs=[x_prev])
            # update x_t
            with torch.no_grad():
                # update guidance scale
                scale_norm = utilso.set_guidance_scale_norm(norm_type=self.scale_norm,
                                                            x_0_hat=x_0_hat.detach(), x_t=x_t.detach(),
                                                            x_prev=x_prev.detach(),
                                                            sample_added_noise=sample_added_noise)
                guidance_scale = scale_norm * self.scale[None, ..., None, None].to(x_prev.device)

                # update x_t
                if self.gradient_x_prev:
                    x_t -= guidance_scale * x_prev.grad
                    gradients = x_prev.grad.cpu()
                else:
                    x_t -= guidance_scale * x_0_hat.grad
                    gradients = x_0_hat.grad.cpu()

        return x_t, sep_loss, variables_dict, gradients


@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        # self.scale = kwargs.get('scale', 1.0)

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


@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
