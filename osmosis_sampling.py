# %% imports
import sys

import numpy as np
from functools import partial
import os
from os.path import join as pjoin
from argparse import ArgumentParser
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


# CONFIG_FILE = r".\configs\osmosis_sample_config.yaml"


# %% main sampling

def main() -> None:
    # read the config file and return an argsparse Namespace object
    args = utilso.arguments_from_file(CONFIG_FILE)
    args.image_size = args.unet_model['image_size']
    args.unet_model['model_path'] = pjoin('.', 'models', args.unet_model['model_path'])

    # Device setting
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    # Prepare dataloader
    data_config = args.data

    # resize small side to be 256px, cropping 256x256, normalizing to [-1,1]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(size=256),
                                    transforms.CenterCrop(size=[256, 256]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # For the case of any data with ground truth (simulation in our case)
    if data_config['ground_truth']:
        gt_flag = True
        dataset = datao.ImagesFolder_GT(root_dir=data_config['root'], gt_rgb_dir=data_config['gt_rgb'],
                                        gt_depth_dir=data_config['gt_depth'], transform=transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)

    # for non ground truth dataset (underwater and haze for our case)
    else:
        gt_flag = False
        dataset = datao.ImagesFolder(data_config['root'], transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)

    # Load unet model
    model = create_model(**args.unet_model)
    model = model.to(device)
    model.eval()

    # extract configurations for args object
    measure_config = args.measurement
    cond_config = args.conditioning
    diffusion_config = args.diffusion
    sample_pattern_config = args.sample_pattern
    aux_loss_config = args.aux_loss

    # output directory
    measurement_name = measure_config['operator']['name']
    out_path = pjoin(".", args.save_dir, measurement_name, args.data['name'])
    out_path = utilso.update_save_dir_date(out_path)

    # create txt file with the configurations
    utilso.yaml_to_txt(CONFIG_FILE, pjoin(out_path, f"configurations.txt"))

    # directory for saving single results
    if args.save_singles:
        save_singles_path = pjoin(out_path, f"single_images")
        os.makedirs(save_singles_path)

    # directory for the results a grid
    if args.save_grids:
        save_grids_path = pjoin(out_path, f"grid_results")
        os.makedirs(save_grids_path)

    # logger
    logger.configure(dir=out_path)
    logger.log(f"pretrained model file: {args.unet_model['model_path']}")

    # when checking the prior the information of the guidance is not relevant
    # if (not args.check_prior) and ("underwater" in args.measurement['operator']['name']):
    if (not args.check_prior):
        log_txt_tmp = utilso.log_text(args=args)
        logger.log(log_txt_tmp)

    # Do Inference, loop over all the images in the dataset
    for i, (ref_img, ref_img_name) in enumerate(loader):

        # in case there is a GT image
        if gt_flag:

            gt_rgb_img = ref_img[1]
            gt_rgb_img_01 = 0.5 * (gt_rgb_img + 1)

            gt_depth_img = ref_img[2]
            gt_depth_img_01 = 0.5 * (gt_depth_img + 1)

            ref_img = ref_img[0]

        # start count time
        start_run_time_ii = datetime.datetime.now()

        # prepare reference image for visualization
        ref_img_01 = 0.5 * (ref_img.detach().cpu()[0] + 1)
        ref_img_name = ref_img_name[0]
        orig_file_name = os.path.splitext(ref_img_name)[0]

        # stop the run before getting to the last image
        if i == args.data['stop_after']:
            break

        # initialize the operator, conditioning and sampler for each image

        # Prepare Operator and noise
        measure_config['operator']['batch_size'] = args.data['batch_size']
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])

        # Prepare conditioning - guidance method
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'],
                                              **sample_pattern_config, **aux_loss_config)
        measurement_cond_fn = cond_method.conditioning

        # Load diffusion sampler and pass the required arguments
        sampler = create_sampler(**diffusion_config)
        # passing the "stable" arguments with the partial method
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
                            pretrain_model=args.unet_model['pretrain_model'], check_prior=args.check_prior,
                            sample_pattern=args.sample_pattern)

        logger.log(f"\nInference image {i}: {ref_img_name}\n")
        ref_img = ref_img.to(device)

        # add noise to the image
        y_n = noiser(ref_img)

        # degamma the input image - use it for haze
        if args.degamma_input:
            y_n_tmp = 0.5 * (y_n + 1)
            y_n = 2 * torch.pow(y_n_tmp, 2.2) - 1

        # Sampling
        x_start_shape = list(ref_img.shape)
        # in case of sampling for osmosis the input model channel is 4 (RGBD)
        x_start_shape[1] = 4 if (args.unet_model["pretrain_model"] == 'osmosis') else x_start_shape[1]

        # sampling noise for the begging of the diffusion model
        if args.sample_pattern['pattern'] == "original":
            global_N = 1
        elif args.sample_pattern['pattern'] == "pcgs":
            global_N = args.sample_pattern['global_N']
        else:
            raise ValueError(f"Unrecognized sample pattern: {args.sample_pattern['pattern']}")

        # loop according the value of global N (from gibbsDDRM)
        for global_ii in range(global_N):

            logger.log(f"global iteration: {global_ii}\n")
            torch.manual_seed(args.manual_seed)

            # the x_T - Gaussian Noise
            x_start = torch.randn(x_start_shape, device=device).requires_grad_()

            # this is the osmosis project additional code
            if args.unet_model["pretrain_model"] == 'osmosis' and not args.check_prior:

                # sampling function which adapted to osmosis project
                sample, variable_dict, loss, out_xstart = sample_fn(x_start=x_start, measurement=y_n,
                                                                    record=args.record_process,
                                                                    save_root=out_path, image_idx=i,
                                                                    record_every=args.record_every,
                                                                    global_iteration=global_ii,
                                                                    original_file_name=orig_file_name,
                                                                    save_grids_path=save_grids_path)

                # output from the network without guidance - split into rgb and depth image
                sample_rgb = out_xstart[0, 0:-1, :, :]
                sample_depth_tmp = out_xstart[0, -1, :, :].unsqueeze(0)
                sample_depth_tmp_rep = sample_depth_tmp.repeat(3, 1, 1)

                # "move" the rgb predicted image to start from 0
                sample_rgb_01 = 0.5 * (sample_rgb + 1)
                sample_rgb_01_clip = torch.clamp(sample_rgb_01, min=0, max=1)

                # "move" the depth predicted image to start from 0
                sample_depth_vis_pmm = utilso.min_max_norm_range_percentile(sample_depth_tmp,
                                                                            vmin=0, vmax=1,
                                                                            percent_low=0.05,
                                                                            percent_high=0.99,
                                                                            is_uint8=False)
                sample_depth_vis_pmm_color = utilso.depth_tensor_to_color_image(sample_depth_vis_pmm)

                # depth for calculations
                sample_depth_calc = utilso.convert_depth(sample_depth_tmp_rep,
                                                         depth_type=args.measurement['operator']['depth_type'],
                                                         value=args.measurement['operator']['value'])

                # phi inf image - relevant for both underwater and haze
                phi_inf = variable_dict['phi_inf'].cpu().squeeze(0)
                phi_inf_image = phi_inf * torch.ones_like(sample_rgb, device=torch.device('cpu'))

                # underwater model
                if 'underwater' in args.measurement['operator']['name']:

                    # create the ingredients for the underwater image
                    phi_a = variable_dict['phi_a'].cpu().squeeze(0)
                    phi_a_image = phi_a * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    phi_b = variable_dict['phi_b'].cpu().squeeze(0)
                    phi_b_image = phi_b * torch.ones_like(sample_rgb, device=torch.device('cpu'))

                    # calculate the underwater parts
                    backscatter_image = phi_inf_image * (1 - torch.exp(-phi_b_image * sample_depth_calc))
                    attenuation_image = torch.exp(-phi_a_image * sample_depth_calc)
                    forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image

                    # calculate norm lost for visualization - degraded_images and ref_img values should be [-1,1]
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round([torch.linalg.norm(
                        degraded_image - ref_img.detach().cpu()).numpy()], decimals=3)

                    # calculate the "clean" image from the predicted phi's and ref image
                    attenuation_flip_image = torch.exp(phi_a_image * sample_depth_calc)
                    sample_rgb_recon = attenuation_flip_image * (ref_img_01 - backscatter_image)

                    # logging values of phi's
                    print_phi_a = [np.round(i, decimals=3) for i in phi_a.cpu().squeeze().tolist()]
                    print_phi_b = [np.round(i, decimals=3) for i in phi_b.cpu().squeeze().tolist()]
                    print_phi_inf = [np.round(i, decimals=3) for i in phi_inf.cpu().squeeze().tolist()]

                    log_value_txt = f"\nInitialized values: " \
                                    f"\nphi_a: [{measure_config['operator']['phi_a']}], lr: {measure_config['operator']['phi_a_eta']}" \
                                    f"\nphi_b: [{measure_config['operator']['phi_b']}], lr: {measure_config['operator']['phi_b_eta']}" \
                                    f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\nphi_a: {print_phi_a}" \
                                    f"\nphi_b: {print_phi_b}" \
                                    f"\nphi_inf: {print_phi_inf}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=3)}"

                    # if GT is existed, check for losses (psnr, ssim, )
                    if gt_flag:
                        psnr_ob = PeakSignalNoiseRatio(data_range=(0., 1.))
                        ssim_ob = StructuralSimilarityIndexMeasure(data_range=(0., 1.))

                        psnr = psnr_ob(sample_rgb_01, gt_rgb_img_01)
                        ssim = ssim_ob(sample_rgb_01, gt_rgb_img_01)

                        add_calc_text = f"\nPSNR: {np.round([psnr], decimals=3)}, SSIM: {np.round([ssim], decimals=3)}\n"
                        log_value_txt += add_calc_text

                    # print phi's on phi_inf image
                    if args.text_on_results:
                        image_text = f"\nInitialized values: " \
                                     f"\nphi_a: [{measure_config['operator']['phi_a']}], lr: {measure_config['operator']['phi_a_eta']}" \
                                     f"\nphi_b: [{measure_config['operator']['phi_b']}], lr: {measure_config['operator']['phi_b_eta']}" \
                                     f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                     f"\nResults values: " \
                                     f"\nphi_a: {print_phi_a}" \
                                     f"\nphi_b: {print_phi_b}" \
                                     f"\nphi_inf: {print_phi_inf}" \
                                     f"\nNorm loss: {np.round(norm_loss_final, decimals=5)}"
                        if gt_flag:
                            image_text += add_calc_text

                        phi_inf_image = utilso.add_text_torch_img(phi_inf_image, image_text, font_size=15)

                # haze model
                elif 'haze' in args.measurement['operator']['name']:

                    # create the ingredients for the hazed image
                    phi_ab = variable_dict['phi_ab'].cpu().squeeze(0)
                    phi_ab_image = phi_ab * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    backscatter_image = phi_inf_image * (1 - torch.exp(-phi_ab_image * sample_depth_calc))
                    attenuation_image = torch.exp(-phi_ab_image * sample_depth_calc)
                    forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image

                    # calculate the "clean" image from the predicted phis, phi_inf and ref image
                    attenuation_flip_image = torch.exp(phi_ab_image * sample_depth_calc)
                    sample_rgb_recon = attenuation_flip_image * (ref_img_01 - backscatter_image)

                    # calculate norm lost for visualization - both degraded_images and ref_img values should be [-1,1]
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round(
                        [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                        decimals=3)

                    # logging values of phi and phi_inf
                    print_phi_ab = np.round(phi_ab.cpu().squeeze(), decimals=3)
                    print_phi_inf = np.round(phi_inf.cpu().squeeze(), decimals=3)
                    log_value_txt = f"\nInitialized values: " \
                                    f"\nphi: [{measure_config['operator']['phi_ab']}], lr: {measure_config['operator']['phi_ab_eta']}" \
                                    f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\nphi: {print_phi_ab}" \
                                    f"\nphi_inf: {print_phi_inf}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"

                    if gt_flag:
                        psnr_ob = PeakSignalNoiseRatio(data_range=(0., 1.))
                        ssim_ob = StructuralSimilarityIndexMeasure(data_range=(0., 1.))

                        psnr = psnr_ob(sample_rgb_01, gt_rgb_img_01)
                        ssim = ssim_ob(sample_rgb_01, gt_rgb_img_01)

                        add_calc_text = f"\nPSNR: {np.round([psnr], decimals=3)}, SSIM: {np.round([ssim], decimals=3)}\n"
                        log_value_txt += add_calc_text

                    # print phis and phi_inf on phi_inf image
                    if args.text_on_results:
                        image_text = f"\nInitialized values: " \
                                     f"\nphi: [{measure_config['operator']['phi_ab']}], lr: {measure_config['operator']['phi_ab_eta']}" \
                                     f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                     f"\nResults values: " \
                                     f"\nphi: {print_phi_ab}" \
                                     f"\nphi_inf: {print_phi_inf}" \
                                     f"\nNorm loss: {np.round(norm_loss_final, decimals=5)}"

                        if gt_flag:
                            image_text += add_calc_text

                        phi_inf_image = utilso.add_text_torch_img(phi_inf_image, image_text, font_size=15)

                    # log results for parameters
                    logger.log(log_value_txt)

                else:
                    raise NotImplementedError("Operator can be for 'underwater' or 'haze' ")

                # saving single images (reference (input), rgb (restored image), depth (depth estimation))
                if args.save_singles:
                    # input - reference image
                    ref_im_pil = tvtf.to_pil_image(ref_img_01)
                    ref_im_pil.save(pjoin(save_singles_path, f'{orig_file_name}_g{global_ii}_ref.png'))

                    # rgb clip - sample_rgb_01_clip
                    sample_rgb_01_clip_pil = tvtf.to_pil_image(sample_rgb_01_clip)
                    sample_rgb_01_clip_pil.save(
                        pjoin(save_singles_path, f'{orig_file_name}_g{global_ii}_rgb.png'))

                    # depth percentile min-max - sample_depth_vis_percentile_norm
                    sample_depth_vis_pmm_color_pil = tvtf.to_pil_image(sample_depth_vis_pmm_color)
                    sample_depth_vis_pmm_color_pil.save(
                        pjoin(save_singles_path, f'{orig_file_name}_g{global_ii}_depth.png'))

                # save extended results in the grid
                if args.save_grids:

                    # additional image can be empty image or the GT image if exists
                    if gt_flag:
                        additional_image = gt_rgb_img_01.squeeze()
                    else:
                        additional_image = torch.zeros_like(sample_rgb_01, device=torch.device('cpu'))

                    # main results visualization
                    grid_list = [ref_img_01, sample_rgb_01_clip, sample_depth_vis_pmm_color, additional_image,
                                 forward_predicted_image, sample_rgb_recon, backscatter_image, phi_inf_image]
                    results_grid = make_grid(grid_list, nrow=4, pad_value=1.)
                    results_grid = utilso.clip_image(results_grid, scale=False, move=False, is_uint8=True) \
                        .permute(1, 2, 0).numpy()
                    results_pil = Image.fromarray(results_grid, mode="RGB")

                    # save the image
                    results_pil.save(pjoin(save_grids_path, f'{orig_file_name}_g{global_ii}_grid.png'))
                    logger.log(f"result images was saved into: {out_path}")

                logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")

            # no osmosis - checking prior
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
                                                                            percent_low=0.05, percent_high=0.95)
                sample_depth_vis_pmm_color = utilso.depth_tensor_to_color_image(sample_depth_vis_pmm)

                # saving seperated images
                if args.save_singles:
                    ref_im_pil = tvtf.to_pil_image(ref_img_01)
                    ref_im_pil.save(pjoin(save_singles_path, f'{orig_file_name}_ref.png'))

                    sample_rgb_pil = tvtf.to_pil_image(sample_rgb_01_clip)
                    sample_rgb_pil.save(pjoin(save_singles_path, f'{orig_file_name}_rgb.png'))

                    sample_depth_vis_pil = tvtf.to_pil_image(sample_depth_vis_pmm_color)
                    sample_depth_vis_pil.save(pjoin(save_singles_path, f'{orig_file_name}_depth.png'))

                # create images grid
                if args.save_grids:
                    grid_list = [ref_img_01, sample_rgb_01_clip, sample_depth_vis_pmm]
                    results_grid = make_grid(grid_list, nrow=3, pad_value=1.)
                    results_grid = utilso.clip_image(results_grid, scale=False, move=False, is_uint8=True)
                    results_pil = tvtf.to_pil_image(results_grid)

                    # save the image
                    results_pil.save(pjoin(save_grids_path, f'{orig_file_name}.png'))
                    logger.log(f"grid results image is saved: {save_grids_path}")

                logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")

    # close the logger txt file
    logger.get_current().close()


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config_file", default="./configs/osmosis_sample_config.yaml",
    parser.add_argument("-c", "--config_file", default="./configs/osmosis_simulation_sample_config.yaml",
                        help="Configurations file")
    args = vars(parser.parse_args())
    CONFIG_FILE = args["config_file"]

    print(f"\nConfiguration file:\n{CONFIG_FILE}\n")

    main()
    print(f"\nFINISH!")
    sys.exit()
