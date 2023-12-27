# %% imports
import sys

import numpy as np
from functools import partial
import os
from os.path import join as pjoin
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont
import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvtf
from torchvision.utils import make_grid
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import matplotlib.cm as cm

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from osmosis_utils import logger

import osmosis_utils.utils as utilso
import osmosis_utils.data as datao

CONFIG_FILE = r"D:\master\debka\dps_pattern\configs\debka_sample_config.yaml"
WANDB = False


# %% main sampling

def main() -> None:
    args = utilso.arguments_from_file(CONFIG_FILE)
    args.image_size = args.unet_model['image_size']

    # Device setting
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    # Prepare dataloader
    data_config = args.data

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(size=256),
                                    transforms.CenterCrop(size=[256, 256]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if data_config['ground_truth'] and ('simulation_type' in data_config['name']):
        gt_flag = True
        water_type = data_config['name'].split("_")[-1]
        dataset = datao.SimulationWaterTypeDataset(root_dir=data_config['root'], water_type=water_type,
                                                   transform=transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)

    elif data_config['ground_truth'] and ('UIEB' in data_config['name']):
        gt_flag = True
        dataset = datao.ImagesFolder_GT(root_dir=data_config['root'], gt_dir=data_config['gt'], transform=transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)

    else:
        gt_flag = False
        dataset = datao.ImagesFolder(data_config['root'], transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)

    # Load model
    model = create_model(**args.unet_model)
    model = model.to(device)
    model.eval()

    # create configurations
    measure_config = args.measurement
    cond_config = args.conditioning
    diffusion_config = args.diffusion
    sample_pattern_config = args.sample_pattern
    quality_loss_config = args.quality_loss

    # Working directory
    out_path = pjoin(args.save_dir, measure_config['operator']['name'], args.data['name'])
    measurement_name = measure_config['operator']['name']

    if "fM" in measure_config['operator']['name']:
        measurement_name += f"_{measure_config['operator']['variance']}"
    out_path = pjoin(args.save_dir, measurement_name, args.data['name'])
    out_path = utilso.update_save_dir_date(out_path)
    # create txt file with the configurations
    utilso.yaml_to_txt(CONFIG_FILE, pjoin(out_path, f"configurations.txt"))
    if args.save_sample:
        os.makedirs(pjoin(out_path, f"images"))

    # logger
    logger.configure(dir=out_path)
    logger.log(f"pretrained model file: {args.unet_model['model_path']}")

    # when checking the prior the information of the guidance is not relevant
    if (not args.check_prior) and ("underwater" in args.measurement['operator']['name']):
        log_txt_tmp = utilso.log_text(args=args)
        logger.log(log_txt_tmp)

    # Do Inference
    for i, (ref_img, ref_img_name) in enumerate(loader):

        if gt_flag:
            gt_img = ref_img[1]
            gt_img_01 = 0.5 * (gt_img + 1)
            ref_img = ref_img[0]

        # start count time
        start_run_time_ii = datetime.datetime.now()

        # for visualization purpose
        ref_img_01 = 0.5 * (ref_img.detach().cpu()[0] + 1)
        ref_img_name = ref_img_name[0]

        # stop the run before getting to the last image
        if i == args.data['stop_after']:
            break

        # incase repeat image is wanted
        if args.repeat_image:
            ref_img = ref_img.repeat(args.repeat_batch_size, 1, 1, 1)

        # initialize the operator, conditioning and sampler for each image

        # Prepare Operator and noise
        measure_config['operator']['batch_size'] = args.data['batch_size']
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])

        # Prepare conditioning method
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'],
                                              **sample_pattern_config, **quality_loss_config)
        measurement_cond_fn = cond_method.conditioning

        # Load diffusion sampler and pass the required arguments
        sampler = create_sampler(**diffusion_config)
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
                            pretrain_model=args.unet_model['pretrain_model'], check_prior=args.check_prior,
                            sample_pattern=args.sample_pattern, guide_and_sample=args.guide_and_sample)

        # Inference
        logger.log(f"\nInference image {i}: {ref_img_name}\n")
        ref_img = ref_img.to(device)

        # add noise to the image
        y_n = noiser(ref_img)

        if args.degamma_input:
            y_n_tmp = 0.5 * (y_n + 1)
            y_n = 2 * torch.pow(y_n_tmp, 2.2) - 1

        # Sampling
        x_start_shape = list(ref_img.shape)
        # in case of sampling for debka the input model channel is 4 (RGBD)
        x_start_shape[1] = 4 if (args.unet_model["pretrain_model"] == 'debka') else x_start_shape[1]

        # sampling noise for the begging of the diffusion model
        if args.sample_pattern['pattern'] == "original":
            global_N = 1
        elif args.sample_pattern['pattern'] == "pcgs":
            global_N = args.sample_pattern['global_N']
        else:
            raise ValueError(f"Unrecognized sample pattern: {args.sample_pattern['pattern']}")

        for global_ii in range(global_N):

            logger.log(f"global iteration: {global_ii}\n")

            torch.manual_seed(args.manual_seed)
            x_start = torch.randn(x_start_shape, device=device).requires_grad_()

            # this is the debka project additional code
            if args.unet_model["pretrain_model"] == 'debka' and not args.check_prior:
                # sampling function which adapted to debka project
                # sample, beta, b_inf, loss, out_xstart
                sample, variable_dict, loss, out_xstart = sample_fn(x_start=x_start, measurement=y_n,
                                                                    record=args.record_process,
                                                                    save_root=out_path, image_idx=i,
                                                                    record_every=args.record_every,
                                                                    global_iteration=global_ii)

                # visualize the last sample from the network without guidance, somehow this looks better
                if args.guide_last_sample:
                    # split into rgb and depth image - not handling results save for a batch of images
                    sample_rgb = sample.detach().cpu()[:, 0:-1, :, :]
                    sample_depth_tmp = sample.detach().cpu()[:, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                else:
                    # output from the network without geuidance
                    sample_rgb = out_xstart[:, 0:-1, :, :]
                    sample_depth_tmp = out_xstart[:, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)

                # "move" the rgb predicted image to start from 0 (the values "sample_rgb" should be between [-1, 1])
                sample_rgb_01 = 0.5 * (sample_rgb + 1)
                sample_rgb_01_clip = torch.clamp(sample_rgb_01, min=0, max=1)
                sample_rgb_01_norm = [utilso.min_max_norm_range(sample_rgb[ii]) for ii in range(sample_rgb_01.shape[0])]
                sample_rgb_01_percentile_norm = [utilso.min_max_norm_range_percentile(sample_rgb[ii],
                                                                                      percent_low=0.01,
                                                                                      percent_high=0.99) for ii in
                                                 range(sample_rgb_01.shape[0])]

                # depth for visualization
                sample_depth_vis_norm = utilso.min_max_norm_range(sample_depth_tmp, vmin=0, vmax=1, is_uint8=False)
                sample_depth_vis_percentile_norm = utilso.min_max_norm_range_percentile(sample_depth_tmp,
                                                                                        vmin=0, vmax=1,
                                                                                        percent_low=0.05,
                                                                                        percent_high=0.99,
                                                                                        is_uint8=False)

                # "move" the depth predicted image to start from 0 (the values "sample_depth" should be between [-1, 1])
                sample_depth_original = 0.5 * (sample_depth_tmp + 1)

                # logging maximum and minimum values of rgb and depth
                log_min_max_txt = f"rgb min: {np.round(sample_rgb.min().numpy(), decimals=3)}, " \
                                  f"rgb max: {np.round(sample_rgb.max().numpy(), decimals=3)}" \
                                  f"\ndepth min: {np.round(sample_depth_tmp.min().numpy(), decimals=3)}, " \
                                  f"depth max: {np.round(sample_depth_tmp.max().numpy(), decimals=3)}"
                logger.log(log_min_max_txt)

                # the case of underwater model
                if ('underwater' in args.measurement['operator']['name']) or (
                        'haze' in args.measurement['operator']['name']):
                    # depth for calculations
                    sample_depth_calc = utilso.convert_depth(sample_depth_tmp,
                                                             depth_type=args.measurement['operator']['depth_type'],
                                                             value=args.measurement['operator']['value'])
                    # b inf image - same color for all image (b inf is a single value for each channel)
                    b_inf = variable_dict['b_inf'].cpu()
                    b_inf_image = b_inf * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    # b_inf_image = b_inf.cpu() * torch.ones_like(sample_rgb, device=torch.device('cpu'))

                    # if the revised model
                    if 'revised' in args.measurement['operator']['name']:

                        # create the ingredients for the underwater image
                        beta_a = variable_dict['beta_a'].cpu()
                        if measure_config['operator']['beta_a_depth']:

                            # beta_a = a * exp(b*depth) + c * exp(d*depth)
                            beta_a_image = beta_a[:, :, 0] * torch.exp(beta_a[:, :, 1] * sample_depth_calc) + \
                                           beta_a[:, :, 2] * torch.exp(beta_a[:, :, 3] * sample_depth_calc)

                            # beta_a = a * depth + b
                            # beta_a_image = beta_a[:, :, 0] * depth_a + exp(self.beta_a[:, :, 1] * depth_a)
                        else:

                            beta_a_image = beta_a * torch.ones_like(sample_rgb, device=torch.device('cpu'))

                        beta_b = variable_dict['beta_b'].cpu()
                        beta_b_image = beta_b * torch.ones_like(sample_rgb, device=torch.device('cpu'))

                        # calculate the underwater parts
                        backscatter_image = b_inf_image * (1 - torch.exp(-beta_b_image * sample_depth_calc))
                        attenuation_image = torch.exp(-beta_a_image * sample_depth_calc)
                        forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image

                        # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                        degraded_image = 2 * forward_predicted_image - 1
                        norm_loss_final = np.round([torch.linalg.norm(degraded_image - ref_img.detach().cpu()).numpy()],
                                                   decimals=3)

                        # calculate the "clean" image from the predicted betas, b_inf and ref image
                        attenuation_flip_image = torch.exp(beta_a_image * sample_depth_calc)
                        sample_rgb_recon = attenuation_flip_image * (ref_img_01.unsqueeze(0) - backscatter_image)

                        # difference image
                        uw_diff = torch.abs(ref_img_01 - forward_predicted_image)

                        # logging values of beta and b_inf
                        print_beta_a = [np.round(i, decimals=3) for i in beta_a.cpu().squeeze().tolist()]
                        print_beta_b = [np.round(i, decimals=3) for i in beta_b.cpu().squeeze().tolist()]
                        print_b_inf = [np.round(i, decimals=3) for i in b_inf.cpu().squeeze().tolist()]
                        log_value_txt = f"\nInitialized values: " \
                                        f"\nbeta_a: [{measure_config['operator']['beta_a']}], lr: {measure_config['operator']['beta_a_eta']}" \
                                        f"\nbeta_b: [{measure_config['operator']['beta_b']}], lr: {measure_config['operator']['beta_b_eta']}" \
                                        f"\nb_inf: [{measure_config['operator']['b_inf']}], lr: {measure_config['operator']['b_inf_eta']}" \
                                        f"\n\nResults values: " \
                                        f"\nbeta_a: {print_beta_a}" \
                                        f"\nbeta_b: {print_beta_b}" \
                                        f"\nb_inf: {print_b_inf}" \
                                        f"\n\nNorm loss: {norm_loss_final}" \
                                        f"\nFinal loss: {np.round(np.array(loss), decimals=3)}"

                        if gt_flag:
                            psnr_ob = PeakSignalNoiseRatio(data_range=(0., 1.))
                            ssim_ob = StructuralSimilarityIndexMeasure(data_range=(0., 1.))

                            psnr = psnr_ob(sample_rgb_01, gt_img_01)
                            ssim = ssim_ob(sample_rgb_01, gt_img_01)

                            add_calc_text = f"\nPSNR: {np.round([psnr], decimals=3)}, SSIM: {np.round([ssim], decimals=3)}\n"
                            log_value_txt += add_calc_text

                        # print betas and b_inf on b_inf image
                        if args.text_on_results:
                            image_text = f"\nInitialized values: " \
                                         f"\nbeta_a: [{measure_config['operator']['beta_a']}], lr: {measure_config['operator']['beta_a_eta']}" \
                                         f"\nbeta_b: [{measure_config['operator']['beta_b']}], lr: {measure_config['operator']['beta_b_eta']}" \
                                         f"\nb_inf: [{measure_config['operator']['b_inf']}], lr: {measure_config['operator']['b_inf_eta']}" \
                                         f"\nResults values: " \
                                         f"\nbeta_a: {print_beta_a}" \
                                         f"\nbeta_b: {print_beta_b}" \
                                         f"\nb_inf: {print_b_inf}" \
                                         f"\nNorm loss: {np.round(norm_loss_final, decimals=5)}"
                            if gt_flag:
                                image_text += add_calc_text
                            b_inf_image = utilso.add_text_torch_img(b_inf_image[0], image_text, font_size=15).unsqueeze(
                                0)

                    # if non revised model
                    else:
                        # create the ingredients for the underwater image
                        beta = variable_dict['beta'].cpu()
                        beta_image = beta * torch.ones_like(sample_rgb, device=torch.device('cpu')).squeeze(0)
                        backscatter_image = b_inf_image * (1 - torch.exp(-beta_image * sample_depth_calc))
                        attenuation_image = torch.exp(-beta_image * sample_depth_calc)
                        forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image

                        # calculate the "clean" image from the predicted betas, b_inf and ref image
                        attenuation_flip_image = torch.exp(beta_image * sample_depth_calc)
                        sample_rgb_recon = attenuation_flip_image * (ref_img_01.unsqueeze(0) - backscatter_image)

                        # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                        degraded_image = 2 * forward_predicted_image - 1
                        norm_loss_final = np.round(
                            [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                            decimals=3)
                        # difference image
                        uw_diff = torch.abs(ref_img_01 - forward_predicted_image)

                        # logging values of beta and b_inf
                        if 'underwater' in args.measurement['operator']['name']:
                            print_beta = [np.round(i, decimals=3) for i in beta.cpu().squeeze().tolist()]
                            print_b_inf = [np.round(i, decimals=3) for i in b_inf.cpu().squeeze().tolist()]
                        else:
                            print_beta = np.round(beta.cpu().squeeze(), decimals=3)
                            print_b_inf = np.round(b_inf.cpu().squeeze(), decimals=3)
                        log_value_txt = f"\nInitialized values: " \
                                        f"\nbeta: [{measure_config['operator']['beta']}], lr: {measure_config['operator']['beta_eta']}" \
                                        f"\nb_inf: [{measure_config['operator']['b_inf']}], lr: {measure_config['operator']['b_inf_eta']}" \
                                        f"\n\nResults values: " \
                                        f"\nbeta: {print_beta}" \
                                        f"\nb_inf: {print_b_inf}" \
                                        f"\n\nNorm loss: {norm_loss_final}" \
                                        f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"

                        if gt_flag:
                            psnr_ob = PeakSignalNoiseRatio(data_range=(0., 1.))
                            ssim_ob = StructuralSimilarityIndexMeasure(data_range=(0., 1.))

                            psnr = psnr_ob(sample_rgb_01, gt_img_01)
                            ssim = ssim_ob(sample_rgb_01, gt_img_01)

                            add_calc_text = f"\nPSNR: {np.round([psnr], decimals=3)}, SSIM: {np.round([ssim], decimals=3)}\n"
                            log_value_txt += add_calc_text

                        # print betas and b_inf on b_inf image
                        if args.text_on_results:
                            image_text = f"\nInitialized values: " \
                                         f"\nbeta: [{measure_config['operator']['beta']}], lr: {measure_config['operator']['beta_eta']}" \
                                         f"\nb_inf: [{measure_config['operator']['b_inf']}], lr: {measure_config['operator']['b_inf_eta']}" \
                                         f"\nResults values: " \
                                         f"\nbeta: {print_beta}" \
                                         f"\nb_inf: {print_b_inf}" \
                                         f"\nNorm loss: {np.round(norm_loss_final, decimals=5)}"

                            if gt_flag:
                                image_text += add_calc_text

                            b_inf_image = utilso.add_text_torch_img(b_inf_image[0], image_text, font_size=15).unsqueeze(
                                0)

                    # # check what is beta - if tensor - it is a non-revised uw model
                    # if isinstance(beta, torch.Tensor):
                    #
                    #     # create the ingredients for the underwater image
                    #     beta_image = beta.cpu().squeeze(0) * torch.ones_like(sample_rgb,
                    #                                                          device=torch.device('cpu')).squeeze(0)
                    #     backscatter_image = b_inf_image * (1 - torch.exp(-beta_image * sample_depth_calc))
                    #     attenuation_image = torch.exp(-beta_image * sample_depth_calc)
                    #     forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image
                    #
                    #     # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                    #     degraded_image = 2 * forward_predicted_image - 1
                    #     norm_loss_final = np.round(
                    #         [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                    #         decimals=3)
                    #     # difference image
                    #     uw_diff = torch.abs(ref_img_01 - forward_predicted_image)
                    #
                    #     # logging values of beta and b_inf
                    #     print_beta = [np.round(i, decimals=3) for i in beta.cpu().squeeze().tolist()]
                    #     print_b_inf = [np.round(i, decimals=3) for i in b_inf.cpu().squeeze().tolist()]
                    #     log_value_txt = f"\nInitialized values: " \
                    #                     f"\nbeta: [{measure_config['operator']['beta']}], coeff: {measure_config['operator']['beta_eta']}" \
                    #                     f"\nb_inf: [{measure_config['operator']['b_inf']}], coeff: {measure_config['operator']['b_inf_eta']}" \
                    #                     f"\n\nResults values: " \
                    #                     f"\nbeta: {print_beta}" \
                    #                     f"\nb_inf: {print_b_inf}" \
                    #                     f"\n\nNorm loss: {norm_loss_final}" \
                    #                     f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"
                    #
                    # # if list - it is a revised uw model and include beta a and beta b
                    # elif len(beta) == 2:
                    #
                    #     # create the ingredients for the underwater image
                    #     beta_a_image = beta[0].cpu() * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    #     beta_b_image = beta[1].cpu() * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    #     backscatter_image = b_inf_image * (1 - torch.exp(-beta_b_image * sample_depth_calc))
                    #     attenuation_image = torch.exp(-beta_a_image * sample_depth_calc)
                    #     forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image
                    #
                    #     # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                    #     degraded_image = 2 * forward_predicted_image - 1
                    #     norm_loss_final = np.round(
                    #         [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                    #         decimals=3)
                    #
                    #     # difference image
                    #     uw_diff = torch.abs(ref_img_01 - forward_predicted_image)
                    #
                    #     # logging values of beta and b_inf
                    #     print_beta_a = [np.round(i, decimals=3) for i in beta[0].cpu().squeeze().tolist()]
                    #     print_beta_b = [np.round(i, decimals=3) for i in beta[1].cpu().squeeze().tolist()]
                    #     print_b_inf = [np.round(i, decimals=3) for i in b_inf.cpu().squeeze().tolist()]
                    #     log_value_txt = f"\nInitialized values: " \
                    #                     f"\nbeta_a: [{measure_config['operator']['beta_a']}], coeff: {measure_config['operator']['beta_a_eta']}" \
                    #                     f"\nbeta_b: [{measure_config['operator']['beta_b']}], coeff: {measure_config['operator']['beta_b_eta']}" \
                    #                     f"\nb_inf: [{measure_config['operator']['b_inf']}], coeff: {measure_config['operator']['b_inf_eta']}" \
                    #                     f"\n\nResults values: " \
                    #                     f"\nbeta_a: {print_beta_a}" \
                    #                     f"\nbeta_b: {print_beta_b}" \
                    #                     f"\nb_inf: {print_b_inf}" \
                    #                     f"\n\nNorm loss: {norm_loss_final}" \
                    #                     f"\nFinal loss: {np.round(np.array(loss), decimals=3)}"
                    #
                    #     # print betas and b_inf on b_inf image
                    #     image_text = f"\nInitialized values: " \
                    #                  f"\nbeta_a: [{measure_config['operator']['beta_a']}], lr: {measure_config['operator']['beta_a_eta']}" \
                    #                  f"\nbeta_b: [{measure_config['operator']['beta_b']}], lr: {measure_config['operator']['beta_b_eta']}" \
                    #                  f"\nb_inf: [{measure_config['operator']['b_inf']}], lr: {measure_config['operator']['b_inf_eta']}" \
                    #                  f"\nResults values: " \
                    #                  f"\nbeta_a: {print_beta_a}" \
                    #                  f"\nbeta_b: {print_beta_b}" \
                    #                  f"\nb_inf: {print_b_inf}" \
                    #                  f"\nNorm loss: {np.round(norm_loss_final, decimals=5)}"
                    #     b_inf_image = utilso.add_text_torch_img(b_inf_image[0], image_text, font_size=15).unsqueeze(0)
                    #
                    # # if something gone wrong
                    # else:
                    #     raise ValueError("Not recognized beta")

                    # log results for parameters
                    logger.log(log_value_txt)

                    # loop over the predicted image - if single only one image will be saved
                    n_images = sample_rgb_01.shape[0]
                    pred_images_list = []
                    for pred_ii in range(n_images):

                        if gt_flag:
                            additional_image = gt_img_01.squeeze()
                        else:
                            additional_image = sample_rgb_01_norm[0]
                            # additional_image = torch.zeros_like(sample_depth_vis[pred_ii])

                        # main results visualization
                        grid_list = [ref_img_01, sample_rgb_01_clip[pred_ii], sample_depth_vis_percentile_norm[pred_ii],
                                     additional_image,
                                     forward_predicted_image[pred_ii], sample_rgb_recon[pred_ii],
                                     backscatter_image[pred_ii], b_inf_image[pred_ii]]
                        results_grid = make_grid(grid_list, nrow=4, pad_value=1.)
                        results_grid = utilso.clip_image(results_grid, scale=False, move=False, is_uint8=True) \
                            .permute(1, 2, 0).numpy()
                        results_pil = Image.fromarray(results_grid, mode="RGB")
                        # save the image
                        results_pil.save(pjoin(out_path, f'image_{i}_{pred_ii}_g{global_ii}.png'))
                        logger.log(f"result images was saved into: {out_path}")

                        # save histograms:
                        histograms_grid_list = [ref_img_01, sample_rgb_01_clip[pred_ii], sample_rgb_01_norm[pred_ii],
                                                utilso.color_histogram(ref_img_01, title='input'),
                                                utilso.color_histogram(sample_rgb_01_clip[pred_ii], title='clip [0,1]'),
                                                utilso.color_histogram(
                                                    utilso.min_max_norm_range(sample_rgb_01_norm[pred_ii]),
                                                    title=f"min-max [{np.round([sample_rgb_01[pred_ii].min()], decimals=2)},"
                                                          f"{np.round([sample_rgb_01[pred_ii].max()], decimals=2)}]")]

                        histograms_grid = make_grid(histograms_grid_list, nrow=3)
                        histograms_grid = utilso.clip_image(histograms_grid, scale=False, move=False, is_uint8=True)
                        tvtf.to_pil_image(histograms_grid).save(
                            pjoin(out_path, f'image_{i}_{pred_ii}_g{global_ii}_hist.png'))

                        if args.save_differences:
                            grid_diff_list = [ref_img_01, forward_predicted_image[pred_ii],
                                              sample_rgb_01[pred_ii], sample_depth_original[pred_ii],
                                              uw_diff[pred_ii],
                                              uw_diff[pred_ii, 0].unsqueeze(0).repeat(3, 1, 1),
                                              uw_diff[pred_ii, 1].unsqueeze(0).repeat(3, 1, 1),
                                              uw_diff[pred_ii, 2].unsqueeze(0).repeat(3, 1, 1)]
                            grid_diff_img = make_grid(grid_diff_list, nrow=4, pad_value=1.)
                            grid_diff_img_pil = tvtf.to_pil_image(grid_diff_img)
                            grid_diff_img_pil.save(pjoin(out_path, f'image_{i}_{pred_ii}_diff_g{global_ii}.png'))

                        # in case of multiple images, we save results of all images on the image
                        if n_images > 1:
                            pred_images_list += [0.5 * (ref_img.detach().cpu()[pred_ii] + 1), sample_rgb_01[pred_ii],
                                                 sample_depth_vis[pred_ii]]

                    # in case of multiple images, we save results of all images on the image
                    if n_images > 1:
                        pred_images_grid = make_grid(pred_images_list, nrow=3, pad_value=1.)
                        pred_images_grid = utilso.clip_image(pred_images_grid, scale=False, move=False, is_uint8=True) \
                            .permute(1, 2, 0).numpy()
                        pred_images = Image.fromarray(pred_images_grid, mode="RGB")
                        pred_images.save(pjoin(out_path, f'image_{i}_summary.png'))

                        # color the std
                        colormap = cm.rainbow

                        # get statistics on the samples in the case of multiple images
                        std_rgb_tmp, mean_rgb = torch.std_mean(input=sample_rgb_01, dim=0)
                        std_rgb = utilso.min_max_norm_range(std_rgb_tmp, vmin=0, vmax=1, is_uint8=False)
                        color_rgb = colormap(std_rgb.squeeze().numpy())
                        std_rgb_r = tvtf.to_tensor(color_rgb[0, :, :, :3])
                        std_rgb_g = tvtf.to_tensor(color_rgb[1, :, :, :3])
                        std_rgb_b = tvtf.to_tensor(color_rgb[2, :, :, :3])

                        std_depth_tmp, mean_depth_tmp = torch.std_mean(input=sample_depth_tmp[:, 0, :, :].unsqueeze(1),
                                                                       dim=0)
                        mean_depth = utilso.min_max_norm_range(mean_depth_tmp, vmin=0, vmax=1, is_uint8=False)
                        mean_depth = mean_depth.repeat(3, 1, 1)
                        std_depth = utilso.min_max_norm_range(std_depth_tmp, vmin=0, vmax=1, is_uint8=False)
                        std_depth = tvtf.to_tensor(colormap(std_depth.squeeze().numpy())[:, :, :3])

                        # add to images list
                        pred_images_stats_list = [0.5 * (ref_img.detach().cpu()[0] + 1), mean_rgb, mean_depth,
                                                  torch.zeros_like(mean_rgb), torch.zeros_like(mean_rgb), std_depth,
                                                  std_rgb_r, std_rgb_g, std_rgb_b]

                        pred_images_stats_grid = make_grid(pred_images_stats_list, nrow=3, pad_value=1.)
                        pred_images_stats_grid = utilso.clip_image(pred_images_stats_grid, scale=False, move=False,
                                                                   is_uint8=True) \
                            .permute(1, 2, 0).numpy()
                        pred_images_stats = Image.fromarray(pred_images_stats_grid, mode="RGB")
                        pred_images_stats.save(pjoin(out_path, f'image_{i}_stats.png'))

                    # saving seperated images
                    if args.save_sample:
                        # original file name
                        orig_file_name = os.path.splitext(ref_img_name)[0]

                        # input - reference image
                        ref_im_pil = tvtf.to_pil_image(ref_img_01)
                        ref_im_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_ref_g{global_ii}.png'))

                        # rgb clip - sample_rgb_01_clip
                        sample_rgb_01_clip_pil = tvtf.to_pil_image(sample_rgb_01_clip[0])
                        sample_rgb_01_clip_pil.save(
                            pjoin(out_path, 'images', f'{orig_file_name}_rgb_g{global_ii}_clip.png'))
                        # rgb min-max - sample_rgb_01_norm
                        sample_rgb_01_mm_pil = tvtf.to_pil_image(sample_rgb_01_norm[0])
                        sample_rgb_01_mm_pil.save(
                            pjoin(out_path, 'images', f'{orig_file_name}_rgb_g{global_ii}_mm.png'))
                        # rgb percentile min-max - sample_rgb_01_percentile_norm
                        sample_rgb_01_percent_mm_pil = tvtf.to_pil_image(sample_rgb_01_percentile_norm[0])
                        sample_rgb_01_percent_mm_pil.save(
                            pjoin(out_path, 'images', f'{orig_file_name}_rgb_g{global_ii}_pmm.png'))

                        # depth min-max - sample_depth_vis_norm
                        sample_depth_vis_mm_pil = tvtf.to_pil_image(sample_depth_vis_norm[0])
                        sample_depth_vis_mm_pil.save(
                            pjoin(out_path, 'images', f'{orig_file_name}_depth_g{global_ii}_mm.png'))
                        # depth percentile min-max - sample_depth_vis_percentile_norm
                        sample_depth_vis_percent_mm_pil = tvtf.to_pil_image(sample_depth_vis_percentile_norm[0])
                        sample_depth_vis_percent_mm_pil.save(
                            pjoin(out_path, 'images', f'{orig_file_name}_depth_g{global_ii}_pmm.png'))

                elif args.measurement['operator']['name'] == 'general_AB':

                    a_mat = variable_dict['a_mat'].cpu()
                    a_mat_vis = utilso.min_max_norm_range(a_mat, vmin=0, vmax=1, is_uint8=False)
                    b_mat = variable_dict['b_mat'].cpu()
                    b_mat_vis = utilso.min_max_norm_range(b_mat, vmin=0, vmax=1, is_uint8=False)

                    forward_predicted_image = a_mat * sample_rgb_01 + b_mat
                    forward_predicted_image_norm = utilso.min_max_norm_range(forward_predicted_image, vmin=0, vmax=1)
                    # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round(
                        [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                        decimals=3)
                    # difference image
                    uw_diff = torch.abs(ref_img_01 - forward_predicted_image)

                    grid_list = [ref_img_01, sample_rgb_01_norm[0], sample_depth_vis_norm[0],
                                 forward_predicted_image_norm[0], a_mat_vis[0], b_mat_vis[0]]
                    grid_image = make_grid(grid_list, nrow=3, pad_value=1.)

                    grid_image_pil = tvtf.to_pil_image(grid_image)
                    grid_image_pil.save(pjoin(out_path, f'image_{i}_{0}.png'))

                    if args.save_differences:
                        grid_diff_list = [ref_img_01, forward_predicted_image[0],
                                          sample_rgb_01[0], sample_depth_vis_norm[0],
                                          uw_diff[0],
                                          uw_diff[0, 0].unsqueeze(0).repeat(3, 1, 1),
                                          uw_diff[0, 1].unsqueeze(0).repeat(3, 1, 1),
                                          uw_diff[0, 2].unsqueeze(0).repeat(3, 1, 1)]
                        grid_diff_img_pil = tvtf.to_pil_image(make_grid(grid_diff_list, nrow=4, pad_value=1.))
                        grid_diff_img_pil.save(pjoin(out_path, f'image_{i}_{0}_diff.png'))

                    grid_list += [a_mat_vis[0, 0].unsqueeze(0).repeat(3, 1, 1),
                                  a_mat_vis[0, 1].unsqueeze(0).repeat(3, 1, 1),
                                  a_mat_vis[0, 2].unsqueeze(0).repeat(3, 1, 1),
                                  b_mat_vis[0, 0].unsqueeze(0).repeat(3, 1, 1),
                                  b_mat_vis[0, 1].unsqueeze(0).repeat(3, 1, 1),
                                  b_mat_vis[0, 2].unsqueeze(0).repeat(3, 1, 1)]

                    grid_image_pil = tvtf.to_pil_image(make_grid(grid_list, nrow=3, pad_value=255))
                    grid_image_pil.save(pjoin(out_path, f'image_{i}_{0}_AB.png'))

                    logger.log(f"result images was saved into: {out_path}")

                    a_mat_mean = np.round(a_mat.squeeze().mean(dim=[1, 2]).numpy(), decimals=3)
                    a_mat_std = np.round(a_mat.squeeze().std(dim=[1, 2]).numpy(), decimals=3)

                    b_mat_mean = np.round(b_mat.squeeze().mean(dim=[1, 2]).numpy(), decimals=3)
                    b_mat_std = np.round(b_mat.squeeze().std(dim=[1, 2]).numpy(), decimals=3)

                    log_value_txt = f"\nInitialized values: " \
                                    f"\na_mat: [{measure_config['operator']['a_mat']}], lr: {measure_config['operator']['a_eta']}" \
                                    f"\nb_mat: [{measure_config['operator']['b_mat']}], lr: {measure_config['operator']['b_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\na_mat: mean: {a_mat_mean}, std: {a_mat_std}" \
                                    f"\nb_mat: mean: {b_mat_mean}, std: {b_mat_std}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"
                    logger.log(log_value_txt)

                    # saving seperated images
                    if args.save_sample:
                        orig_file_name = os.path.splitext(ref_img_name)[0]

                        ref_im_pil = tvtf.to_pil_image(ref_img_01)
                        ref_im_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_ref.png'))

                        sample_rgb_pil = tvtf.to_pil_image(sample_rgb_01_clip[0])
                        sample_rgb_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_rgb.png'))

                        sample_depth_vis_pil = tvtf.to_pil_image(sample_depth_vis_norm[0])
                        sample_depth_vis_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_depth_mm.png'))

                        sample_depth_vis_pmm_pil = tvtf.to_pil_image(sample_depth_vis_percentile_norm[0])
                        sample_depth_vis_pmm_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_depth_pmm.png'))


                elif args.measurement['operator']['name'] == 'general_fM':

                    f_factor = variable_dict['f_factor'].cpu()
                    f_factor_vis = f_factor * torch.ones_like(sample_rgb, device=torch.device("cpu"))

                    m_mat = variable_dict['m_mat'].cpu()
                    m_mat_vis = utilso.min_max_norm_range(m_mat, vmin=0, vmax=1, is_uint8=False)

                    if args.measurement['operator']['variance'] == 'original':
                        forward_predicted_image = f_factor * sample_rgb_01 + m_mat
                    elif args.measurement['operator']['variance'] == 'transmission':
                        forward_predicted_image = m_mat * sample_rgb_01 + f_factor * (1 - m_mat)
                    else:
                        raise ValueError

                    # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round(
                        [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                        decimals=3)
                    # difference image
                    uw_diff = torch.abs(ref_img_01 - forward_predicted_image)

                    grid_list = [ref_img_01, sample_rgb_01[0], sample_depth_vis[0],
                                 forward_predicted_image[0], f_factor_vis[0], m_mat_vis[0]]
                    grid_image_pil = tvtf.to_pil_image(make_grid(grid_list, nrow=3, pad_value=255))
                    grid_image_pil.save(pjoin(out_path, f'image_{i}_{0}.png'))

                    if args.save_differences:
                        grid_diff_list = [ref_img_01, forward_predicted_image[0],
                                          sample_rgb_01[0], sample_depth_vis[0],
                                          uw_diff[0],
                                          uw_diff[0, 0].unsqueeze(0).repeat(3, 1, 1),
                                          uw_diff[0, 1].unsqueeze(0).repeat(3, 1, 1),
                                          uw_diff[0, 2].unsqueeze(0).repeat(3, 1, 1)]
                        grid_diff_img_pil = tvtf.to_pil_image(make_grid(grid_diff_list, nrow=4, pad_value=1.))
                        grid_diff_img_pil.save(pjoin(out_path, f'image_{i}_{0}_diff.png'))

                    grid_list += [m_mat_vis[0, 0].unsqueeze(0).repeat(3, 1, 1),
                                  m_mat_vis[0, 1].unsqueeze(0).repeat(3, 1, 1),
                                  m_mat_vis[0, 2].unsqueeze(0).repeat(3, 1, 1)]

                    grid_image_pil = tvtf.to_pil_image(make_grid(grid_list, nrow=3, pad_value=255))
                    grid_image_pil.save(pjoin(out_path, f'image_{i}_{0}_fm.png'))

                    logger.log(f"result images was saved into: {out_path}")

                    m_mat_mean = np.round(m_mat.squeeze().mean(dim=[1, 2]).numpy(), decimals=3)
                    m_mat_std = np.round(m_mat.squeeze().std(dim=[1, 2]).numpy(), decimals=3)

                    log_value_txt = f"\nInitialized values: " \
                                    f"\nf_factor: [{measure_config['operator']['f_factor']}], lr: {measure_config['operator']['f_eta']}" \
                                    f"\nm_mat: [{measure_config['operator']['m_mat']}], lr: {measure_config['operator']['m_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\nf_factor: {np.round(f_factor.squeeze().tolist(), decimals=3)}" \
                                    f"\nm_mat: mean: {m_mat_mean}, std: {m_mat_std}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"
                    logger.log(log_value_txt)

                    # saving seperated images
                    if args.save_sample:
                        orig_file_name = os.path.splitext(ref_img_name)[0]
                        ref_im_pil = tvtf.to_pil_image(ref_img_01[0])
                        ref_im_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_ref.png'))

                        sample_rgb_pil = tvtf.to_pil_image(sample_rgb_01[0])
                        sample_rgb_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_rgb.png'))

                        sample_depth_vis_pil = tvtf.to_pil_image(sample_depth_vis[0])
                        sample_depth_vis_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_depth.png'))

                else:
                    raise NotImplementedError
                logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")

            # no debka - checking prior
            else:
                sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

                # split into rgb and depth image - not handling results save for a batch of images
                sample_rgb = sample.cpu()[0, 0:-1, :, :]
                sample_depth_tmp = sample.cpu()[0, -1, :, :].repeat(3, 1, 1)

                # "move" the rgb predicted image to start from 0 (the values "sample_rgb" should be between [-1, 1])
                sample_rgb_01 = 0.5 * (sample_rgb + 1)
                sample_rgb_01_clip = torch.clamp(sample_rgb_01, 0., 1.)

                # used for visualization
                sample_depth_vis = utilso.min_max_norm_range(sample_depth_tmp, vmin=0, vmax=1, is_uint8=False)
                sample_depth_vis_pmm = utilso.min_max_norm_range_percentile(sample_depth_tmp,
                                                                            percent_low=0.01, percent_high=0.99)

                # saving seperated images
                if args.save_sample:
                    orig_file_name = os.path.splitext(ref_img_name)[0]

                    ref_im_pil = tvtf.to_pil_image(ref_img_01)
                    ref_im_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_ref.png'))

                    sample_rgb_pil = tvtf.to_pil_image(sample_rgb_01_clip)
                    sample_rgb_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_rgb.png'))

                    sample_depth_vis_pil = tvtf.to_pil_image(sample_depth_vis_pmm)
                    sample_depth_vis_pil.save(pjoin(out_path, 'images', f'{orig_file_name}_depth.png'))

                # create images grid
                grid_list = [ref_img_01, sample_rgb_01_clip, sample_depth_vis_pmm]
                results_grid = make_grid(grid_list, nrow=3, pad_value=1.)
                results_grid = utilso.clip_image(results_grid, scale=False, move=False, is_uint8=True)
                results_pil = tvtf.to_pil_image(results_grid)

                # save the image
                results_pil.save(pjoin(out_path, f'{orig_file_name}.png'))
                logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")
                logger.log(f"result image was saved: {out_path}")

    # close the logger txt file
    logger.get_current().close()


if __name__ == '__main__':
    main()
    print(f"\nFINISH!")
    sys.exit()
