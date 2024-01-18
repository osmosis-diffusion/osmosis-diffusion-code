# %% imports
import sys

import numpy as np
import os
from os.path import join as pjoin
from argparse import ArgumentParser
from PIL import Image
import datetime

import torch
import torchvision.transforms.functional as tvtf
from torchvision.utils import make_grid

from osmosis_utils import logger
import osmosis_utils.utils as utilso

from guided_diffusion.unet import UNetModel
from osmosis_utils.diffusion import GaussianDiffusion


# %% main sampling

def main() -> None:
    # read the config file and return an argsparse Namespace object
    args = utilso.arguments_from_file(CONFIG_FILE)
    args.image_size = args.unet_model['image_size']
    args.unet_model['model_path'] = pjoin('.', 'models', args.unet_model['model_path'])

    # Device setting
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    # output directory
    out_path = utilso.update_save_dir_date(pjoin(".", args.save_dir, "RGBD_prior"))

    # create txt file with the configurations
    utilso.yaml_to_txt(CONFIG_FILE, pjoin(out_path, f"configurations.txt"))

    # directory for saving single results
    if args.save_singles:
        save_singles_path = pjoin(out_path, f"single_images")
        os.makedirs(save_singles_path)
    else:
        save_singles_path = None

    # directory for the results a grid
    if args.save_grids:
        save_grids_path = pjoin(out_path, f"grid_results")
        os.makedirs(save_grids_path)
    else:
        save_grids_path = None

    # logger
    logger.configure(dir=out_path)
    logger.log(f"pretrained model file: {args.unet_model['model_path']}")

    # diffusion UNet model
    diff_unet = UNetModel(image_size=256, in_channels=3, out_channels=6,
                          model_channels=256, num_res_blocks=2, channel_mult=(1, 1, 2, 2, 4, 4),
                          attention_resolutions=[32, 16, 8], num_head_channels=64, dropout=0.1,
                          resblock_updown=True, use_scale_shift_norm=True)

    # in the case of osmosis - rgbd prior, the input and output layers of the unet will have 4 channels (RGBD)
    if args.unet_model['pretrain_model'] == 'osmosis':
        diff_unet = utilso.change_input_output_unet(model=diff_unet, in_channels=4, out_channels=8)

    # checkpoint - loading pretrained model
    diff_unet.load_state_dict(torch.load(args.unet_model['model_path'], map_location='cpu'))
    diff_unet.to(device)

    # evaluation mode
    diff_unet.eval()

    # in case of sampling from rgbd prior the input model channel is 4 (RGBD)
    x_start_dim = 4 if (args.unet_model["pretrain_model"] == 'osmosis') else 3
    torch.manual_seed(args.manual_seed)

    # Do Inference, loop over all the images in the dataset
    for im_idx in range(args.number_of_images):

        # start count time
        start_run_time_ii = datetime.datetime.now()
        logger.log(f"\nInference image {im_idx}/{args.number_of_images}\n")

        # Diffusion process
        diffusion = GaussianDiffusion(T=args.diffusion['steps'], schedule=args.diffusion['noise_schedule'])

        # sample by inverse the diffusion model
        x = diffusion.inverse(net=diff_unet, shape=(x_start_dim, args.image_size, args.image_size),
                              image_channels=x_start_dim, steps=args.diffusion['timestep_respacing'], device=device,
                              record_process=args.record_process, record_every=args.record_every,
                              save_path=save_grids_path, image_idx=im_idx)

        # split into RGB image and Depth image
        x = x.cpu()
        x_rgb = 0.5 * (1 + x[:, 0:3, :, :].squeeze())
        if x_start_dim == 4:
            x_d = x[:, 3, :, :]
            x_d_pmm = utilso.min_max_norm_range_percentile(x_d, percent_low=0.05, percent_high=0.99)
            x_d_pmm_color = utilso.depth_tensor_to_color_image(x_d_pmm)

        # save the images
        if args.save_singles:
            tvtf.to_pil_image(x_rgb).save(pjoin(save_singles_path, f"image_{im_idx}_rgb.png"))
            if x_start_dim == 4:
                tvtf.to_pil_image(x_d_pmm_color).save(pjoin(save_singles_path, f"image_{im_idx}_depth.png"))

        # save the images as pairs
        if args.save_grids and x_start_dim == 4:
            grid_list = [x_rgb, x_d_pmm_color]
            grid_image = make_grid(grid_list, pad_value=1.)
            tvtf.to_pil_image(grid_image).save(pjoin(save_grids_path, f"image_{im_idx}.png"))

        logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")


# close the logger txt file
logger.get_current().close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./configs/RGBD_sample_config.yaml",
                        help="Configurations file")
    args = vars(parser.parse_args())
    CONFIG_FILE = args["config_file"]

    print(f"\nConfiguration file:\n{CONFIG_FILE}\n")

    main()
    print(f"\nFINISH!")
    sys.exit()
