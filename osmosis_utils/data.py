import cv2
import os
from os.path import join as pjoin
import warnings

import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
from natsort import natsorted
import h5py
import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvtf

import osmosis_utils.utils as utilso


# from osmosis_utils.custom_path import CustomPath


# %% Scannet++ Dataset

class ScannetppDataset(Dataset):
    """Scannetpp dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform
        self.images_list = natsorted((self.root_dir.glob("*/iphone/rgb/*.jpg")))
        self.depth_list = natsorted((self.root_dir.glob("*/iphone/depth/*.png")))
        assert len(self.images_list) == len(self.depth_list)
        print(f"Found {len(self.images_list)} samples")

        # Scannet++ depth input is in the range[??, ??]
        self.scannetpp_range = [0, 10]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])  # PIL image
        # numpy array uint16, mm, by https://github.com/scannetpp/scannetpp#render-depth-for-dslr-and-iphone
        depth_map_tmp = cv2.imread(str(self.depth_list[idx]), cv2.IMREAD_UNCHANGED)

        min_size = min(depth_map_tmp.shape[0:2])

        # torch tensor float32 in range [0, 1], shape (3, H, W)
        image = tvtf.to_tensor(image)
        img = tvtf.normalize(tvtf.center_crop(tvtf.resize(image, size=self.image_size),
                                              output_size=[self.image_size, self.image_size]), [0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
        # torch tensor float32 in range [0, 2 ** 16 - 1], shape (1, H', W')
        depth_map = torch.tensor(depth_map_tmp.astype(np.float32)).unsqueeze(0)
        # depth_map = tvtf.to_tensor(depth_map.astype(np.float32))

        depth_map = tvtf.center_crop(
            tvtf.resize(depth_map, size=self.image_size),
            output_size=[self.image_size, self.image_size])

        # the input is in mm - change to meters
        depth_map = depth_map / 1000.
        depth_mask = (depth_map > 0).long().to(torch.float32)

        # normalize according the maximum value of the dataset
        depth_map = depth_map / self.scannetpp_range[1]
        depth_map = torch.clamp(depth_map, 0., 1.)
        # then normalize between -1 to 1
        depth_map = 2 * depth_map - 1

        # concatenate the rgb and the depth
        rgbd = torch.cat([img, depth_map], dim=0)

        if self.transform:
            rgbd = self.transform(rgbd)
            depth_mask = self.transform(depth_mask)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% DIODE Dataset
class DIODEDataset(Dataset):
    """DIODE dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform
        self.images_list = natsorted(glob.glob(os.path.join(self.root_dir, '**', '*.png'), recursive=True))
        self.depth_list = natsorted(glob.glob(os.path.join(self.root_dir, '**', '*_depth.npy'), recursive=True))
        self.depth_mask_list = natsorted(
            glob.glob(os.path.join(self.root_dir, '**', '*_depth_mask.npy'), recursive=True))

        # DOIDE depth input is in the range[0.6, 350]
        self.diode_range = [0.6, 350]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        # TODO:
        #   DOIDE depth input is in the range [0.6,350]
        #   0 value in the depth map means that no information (the sky and more areas which probably not known)
        #   in the monocular DepthGen, they ran sky-segmentor: https://github.com/google/sky-optimization (there is no code!!!!)
        #   "color" it with the maximum value (80m in their case) and interpolate by nearest neighbour missing pixels
        #   our problem that in segmenting the sky
        #   here is a c++ code here: https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing
        #   check with shlomi amitai, he asked questions there

        image = Image.open(self.images_list[idx])
        depth_map = np.load(self.depth_list[idx])
        depth_mask = np.load(self.depth_mask_list[idx])
        min_size = min(depth_map.shape[0:2])

        img = tvtf.normalize(
            tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(image), size=[min_size, min_size]),
                output_size=[self.image_size, self.image_size]),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.disparity:
            # TODO: fix disparity as in HR-WSI and ReDWeb
            depth_map = tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(1 / (depth_map + 1e-4)), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size])
            raise NotImplementedError
        else:

            depth_map = tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(depth_map), size=[min_size, min_size]),
                output_size=[self.image_size, self.image_size])
            # normalize according the maximum value of the dataset
            depth_map = depth_map / self.diode_range[1]
            # then normalize between -1 to 1
            depth_map = 2 * depth_map - 1

            depth_mask = tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(depth_mask), size=[min_size, min_size]),
                output_size=[self.image_size, self.image_size])

            rgbd = torch.cat([img, depth_map], dim=0)

        if self.transform:
            rgbd = self.transform(rgbd)
            depth_mask = self.transform(depth_mask)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

            # return rgbd, depth_mask

        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% DIODE indoor Dataset
class DIODEDatasetIndoor(Dataset):
    """DIODE dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform
        self.images_list = natsorted(glob.glob(os.path.join(self.root_dir, '**', '*.png'), recursive=True))
        self.depth_list = natsorted(glob.glob(os.path.join(self.root_dir, '**', '*_depth.npy'), recursive=True))
        self.depth_mask_list = natsorted(
            glob.glob(os.path.join(self.root_dir, '**', '*_depth_mask.npy'), recursive=True))

        # DOIDE depth input is in the range[0.6, 350]
        self.diode_range = [0.6, 10.]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image = Image.open(self.images_list[idx])
        depth_map = np.load(self.depth_list[idx])
        depth_mask = np.load(self.depth_mask_list[idx])
        min_size = min(depth_map.shape[0:2])

        img = tvtf.normalize(
            tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(image), size=self.image_size),
                output_size=[self.image_size, self.image_size]),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.disparity:
            # TODO: fix disparity as in HR-WSI and ReDWeb
            depth_map = tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(1 / (depth_map + 1e-4)), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size])
            raise NotImplementedError
        else:

            depth_map = tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(depth_map), size=self.image_size),
                output_size=[self.image_size, self.image_size])
            # normalize according the maximum value of the dataset
            depth_map = depth_map / self.diode_range[1]
            depth_map = torch.clamp(depth_map, min=0., max=1.)
            # then normalize between -1 to 1
            depth_map = 2 * depth_map - 1

            depth_mask = tvtf.center_crop(
                tvtf.resize(tvtf.to_tensor(depth_mask), size=self.image_size),
                output_size=[self.image_size, self.image_size])

            rgbd = torch.cat([img, depth_map], dim=0)

        if self.transform:
            rgbd = self.transform(rgbd)
            depth_mask = self.transform(depth_mask)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

            # return rgbd, depth_mask

        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% REDWEB-S Dataset
class REDWEBDataset(Dataset):
    """REDWEB dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform
        self.images_list = natsorted(glob.glob(os.path.join(self.root_dir, 'RGB', '*.jpg'), recursive=True))
        self.depth_list = natsorted(glob.glob(os.path.join(self.root_dir, 'depth', '*.png'), recursive=True))

        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image = Image.open(self.images_list[idx])
        depth_map = Image.open(self.depth_list[idx])
        min_size = min(depth_map.size)
        img = tvtf.normalize(
            tvtf.resize(
                tvtf.center_crop(
                    tvtf.to_tensor(image), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size]),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.disparity:
            # TODO: fix the bug of the convertion between disparity and depth map
            depth_map = tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(1 / (depth_map + 1e-4)), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size])
        else:
            depth_map = tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(depth_map), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size])

        depth_map = tvtf.normalize(depth_map, [0.5], [0.5])

        rgbd = torch.cat([img, depth_map], dim=0)
        depth_mask = torch.ones_like(depth_map)

        if self.transform:
            rgbd = self.transform(rgbd)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

        # return rgbd, depth_mask
        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% HRWSI Dataset
class HRWSIDataset(Dataset):
    """HRWSI dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform
        self.images_list = natsorted(glob.glob(os.path.join(self.root_dir, 'imgs', '*.jpg'), recursive=True))
        self.depth_list = natsorted(glob.glob(os.path.join(self.root_dir, 'gts', '*.png'), recursive=True))
        self.depth_mask_list = natsorted(glob.glob(os.path.join(self.root_dir, 'valid_masks', '*.png'), recursive=True))

        # input depth is disparity - how to handle it
        self.one_over_disparity_convert = False

        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_pil = Image.open(self.images_list[idx])
        depth_map_pil = Image.open(self.depth_list[idx])
        depth_mask_pil = Image.open(self.depth_mask_list[idx])
        min_size = min(depth_map_pil.size)

        # handling RGB image
        img = tvtf.normalize(
            tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(image_pil), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size]),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # handling depth map image
        if self.disparity:
            depth_map = tvtf.resize(
                tvtf.center_crop(tvtf.to_tensor(depth_map_pil), output_size=[min_size, min_size]),
                size=[self.image_size, self.image_size])
        else:

            # TODO:
            #   depth images are disparity: the sky are 0. but the values are uint8, therefore - not scaled?
            #   **** what do we want to do? *****
            #       1. in the case of [depth = 1/disparity] we get big gap between the sky and rest of the depth image
            #           which is not that good for osmosis_utils opinion
            #           option 1  - give the sky other value which is closer to the other values - "max_distance_value"
            #       2. option 2 -  maybe use [depth = 255-disparity]. it is geometric incorrect (deborah?)

            depth_map = tvtf.to_tensor(depth_map_pil)

            # original depth map is disparity, therefore flipping is required
            # addition normaliztion is required since the flip changed the value
            if self.one_over_disparity_convert:
                depth_map = utilso.min_max_norm(img=1 / (depth_map + 1e-5), is_uint8=False)
            else:
                depth_map = 1. - depth_map

            # crop and resize
            depth_map = tvtf.resize(tvtf.center_crop(depth_map, output_size=[min_size, min_size]),
                                    size=[self.image_size, self.image_size])

        # normalize
        depth_map = tvtf.normalize(depth_map, [0.5], [0.5])

        # handling depth mask
        depth_mask = tvtf.resize(
            tvtf.center_crop(tvtf.to_tensor(depth_mask_pil), output_size=[min_size, min_size]),
            size=[self.image_size, self.image_size])

        rgbd = torch.cat([img, depth_map], dim=0)

        if self.transform:
            rgbd = self.transform(rgbd)
            depth_mask = self.transform(depth_map)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

        # return rgbd, depth_mask
        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% KITTI Dataset

class KITTIDataset(Dataset):
    """KITTI dataset."""

    def __init__(self, root_dir, image_size=256, disparity=False, transform=None, random_flip=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            image_size (int) : Input size to the diffusion network
            disparity (bool) : Use disparity instead of depth
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.disparity = disparity
        self.transform = transform

        # define a random crop transformation, since the kitti dataset the images are "long" horizontally
        self.random_crop = transforms.RandomCrop(size=(self.image_size, self.image_size))

        # TODO: paths we be changed according the data that we will work with

        self.images_list = natsorted(
            glob.glob(os.path.join(self.root_dir, "**", "image_02", "data", "*.png"), recursive=True))
        self.depth_list = natsorted(glob.glob(os.path.join(self.root_dir, "**", "image_02", "data", "results",
                                                           "depth_images_norm", "*.png"), recursive=True))
        self.depth_mask_list = natsorted(glob.glob(os.path.join(self.root_dir, "**", "image_02", "data", "results",
                                                                "valid_masks", "*.mat"), recursive=True))

        # check dissimilarities in the case of missing drive directories
        # for now - I assume that there are missing depth maps and not missing images

        # check existence of drive directories
        images_drive = set([ii.split(os.sep)[-4] for ii in self.images_list])
        depth_drive = set([ii.split(os.sep)[-6] for ii in self.depth_list])
        depth_mask_drive = set([ii.split(os.sep)[-6] for ii in self.depth_mask_list])

        # differences
        image_depth = images_drive - depth_drive
        image_mask = images_drive - depth_mask_drive

        # check if they are the same, they should be the same
        assert image_depth == image_mask, \
            "something is wrong - dissimilarities between depths images and masks drives"

        # check if they empty or not
        if bool(image_depth):  # True means not empty
            # remove the extras
            extras_list = list(image_depth)
            self.images_list = [ii for ii in self.images_list if not (ii.split(os.sep)[-4] in extras_list)]

        # remove missing images/depths
        # for now - I assume that there are missing depth maps and not missing images
        images_drive = natsorted(list(set([ii.split(os.sep)[-4] for ii in self.images_list])))
        depth_drive = natsorted(list(set([ii.split(os.sep)[-6] for ii in self.depth_list])))

        # check if they are the same, they should be the same
        assert set(images_drive) == set(images_drive), \
            "something is wrong - dissimilarities between images and depths drives"

        # loop over the drives directories
        for drive in depth_drive:

            # get current drive
            images_drive_cur = [ii for ii in self.images_list if drive in ii]
            depth_drive_cur = [ii for ii in self.depth_list if drive in ii]

            # get images numbers
            images_drive_cur_numbers = [int(os.path.splitext(os.path.split(a)[-1])[0]) for a in images_drive_cur]
            depth_drive_cur_numbers = [int(os.path.splitext(os.path.split(a)[-1])[0].split(sep="_")[-1]) for a in
                                       depth_drive_cur]

            # get dissimilarities
            images_depth_files = set(images_drive_cur_numbers) - set(depth_drive_cur_numbers)
            depth_images_files = set(depth_drive_cur_numbers) - set(images_drive_cur_numbers)

            # remove from images list in case there is a dissimilarities
            if bool(images_depth_files):
                images_depth_files = list(images_depth_files)
                idx_to_remove = [images_drive_cur_numbers.index(ii) for ii in images_depth_files]
                paths_to_remove = [images_drive_cur[ii] for ii in idx_to_remove]
                _ = [self.images_list.remove(ii) for ii in paths_to_remove]

            # assume that there are missing depth maps and not missing images
            assert not bool(depth_images_files), "assume that there are missing depth maps and not missing images"

        # sort lists again
        self.images_list = natsorted(self.images_list)
        self.depth_list = natsorted(self.depth_list)
        self.depth_mask_list = natsorted(self.depth_mask_list)

        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        # read the data
        image = Image.open(self.images_list[idx])
        depth_map = Image.open(self.depth_list[idx])
        depth_mask = loadmat(self.depth_mask_list[idx])['valid_mask']
        # depth_mask = loadmat(self.depth_mask_list[idx])['valid_raw_mask']

        # option 1 - first resize and secod crop
        # img = tvtf.normalize(tvtf.resize(tvtf.to_tensor(image), size=self.image_size), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # depth_mask = tvtf.resize(tvtf.to_tensor(255 * depth_mask), size=self.image_size,
        #                          interpolation=transforms.InterpolationMode.NEAREST)

        # option 1 - since usually the upper area is not relevant (no lidar information)
        #            so not resize, only crop
        img_tensor = tvtf.to_tensor(image)
        depth_mask_tensor = tvtf.to_tensor(255 * depth_mask)

        img = tvtf.normalize(
            tvtf.crop(img_tensor, top=img_tensor.shape[1] - self.image_size,
                      left=0,
                      height=self.image_size,
                      width=img_tensor.shape[2]), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        depth_mask = tvtf.crop(depth_mask_tensor, top=depth_mask_tensor.shape[1] - self.image_size,
                               left=0,
                               height=self.image_size,
                               width=depth_mask_tensor.shape[2])

        if self.disparity:
            NotImplementedError()
            # depth_map = 255 - np.array(depth_map)
            # depth_map = tvtf.resize(tvtf.to_tensor(depth_map), size=self.image_size)
        else:
            # depth_map = tvtf.resize(tvtf.to_tensor(depth_map), size=self.image_size)
            depth_map_tensor = tvtf.to_tensor(depth_map)
            depth_map = tvtf.crop(depth_map_tensor, top=depth_map_tensor.shape[1] - self.image_size,
                                  left=0,
                                  height=self.image_size,
                                  width=depth_map_tensor.shape[2])

        depth_map = tvtf.normalize(depth_map, mean=[0.5], std=[0.5])
        rgbd = torch.cat([img, depth_map], dim=0)

        # random crop the rgbd and the mask with the same randomized crop parameters
        crop_params = self.random_crop.get_params(rgbd, output_size=(self.image_size, self.image_size))
        rgbd = tvtf.crop(rgbd, *crop_params)
        depth_mask = tvtf.crop(depth_mask, *crop_params)

        if self.transform:
            rgbd = self.transform(rgbd)
            depth_mask = self.transform(depth_mask)

        if self.random_flip:
            # horizontal flip
            if random.random() > 0.5:
                rgbd = tvtf.hflip(rgbd)
                depth_mask = tvtf.hflip(depth_mask)

            # vertical_flip
            if random.random() > 0.5:
                rgbd = tvtf.vflip(rgbd)
                depth_mask = tvtf.vflip(depth_mask)

        # return rgbd, depth_mask
        return {"rgbd": rgbd, 'mask_valid': depth_mask}


# %% Load data function - integrate RGBD datasets for
a = 10

datasets_map = {"DIODE": DIODEDataset,
                "DIODEindoor": DIODEDatasetIndoor,
                "ReDWeb": REDWEBDataset,
                "HRWSI": HRWSIDataset,
                "KITTI": KITTIDataset,
                "ScanNetpp": ScannetppDataset}


def load_multi_dataset(datasets_name_list: list, datasets_paths_list: list):
    """
    concatenation of datasets
    in case of using multiple dataset

    :param datasets_name_list: names of the datasets
    :param datasets_paths_list: path for the datasets directories
    :return: pytorch ConcatDataset object
    """

    datasets_list = [datasets_map[dataset_name_ii](path_ii, random_flip=True) for dataset_name_ii, path_ii in
                     zip(datasets_name_list, datasets_paths_list)]

    return ConcatDataset(datasets_list)


def create_dataloader(datasets_name_list: list, datasets_paths_list: list, batch_size: int, shuffle: bool, workers=0):
    datasets_concat = load_multi_dataset(datasets_name_list, datasets_paths_list)
    dataloader = DataLoader(datasets_concat, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

    while True:
        yield from dataloader


# %% ImageFolder Dataset

class ImagesFolder(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_list = natsorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        try:
            image = Image.open(os.path.join(self.root_dir, self.images_list[idx]))
        except:
            print("\n**************\nexpect\n**************\n")
            image = cv2.imread(os.path.join(self.root_dir, self.images_list[idx]), cv2.IMREAD_UNCHANGED)
            image = image // 255

        if self.transform is not None:
            image = self.transform(image)

        return image, self.images_list[idx]


# %% ImageFolder Dataset with gt (simulation)

class ImagesFolder_GT_results(Dataset):

    def __init__(self, gt_dir, results_dir, transform=None):
        self.gt_dir = gt_dir
        self.results_dir = results_dir

        self.gt_list = natsorted(glob.glob(pjoin(gt_dir, "*.*")))
        self.simulate_list = natsorted(glob.glob(pjoin(results_dir, "*ref.png")))
        self.rgb_list = natsorted(glob.glob(pjoin(results_dir, "*rgb.png")))
        self.depth_list = natsorted(glob.glob(pjoin(results_dir, "*depth.png")))
        self.transform = transform

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        image_name = os.path.splitext(os.path.basename(self.gt_list[idx]))[0]
        gt = Image.open(self.gt_list[idx])
        simulate = Image.open(self.simulate_list[idx])
        rgb = Image.open(self.rgb_list[idx])
        depth = Image.open(self.depth_list[idx])

        if self.transform is not None:
            gt = self.transform(gt)
            simulate = self.transform(simulate)
            rgb = self.transform(rgb)
            depth = self.transform(depth)

        return gt, simulate, rgb, depth, image_name


# %% ImageFolder Dataset with gt
class ImagesFolder_GT(Dataset):

    def __init__(self, root_dir, gt_dir, transform=None):
        self.gt_dir = gt_dir
        self.root_dir = root_dir

        self.gt_list = natsorted(glob.glob(pjoin(gt_dir, "*.*")))
        self.images_list = natsorted(glob.glob(pjoin(root_dir, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.images_list[idx])
        image = Image.open(self.images_list[idx])
        gt_image = Image.open(self.gt_list[idx])

        if self.transform is not None:
            gt_image = self.transform(gt_image)
            image = self.transform(image)

        return [image, gt_image], image_name


# %% dataset from paths


# %% NYU dataset class from the sub mat file

class NyuDataset(Dataset):
    def __init__(self, root, topk_labels=None, transform=None, normalize=False, depth_norm=10):
        """
            NYU Depth Dataset
            root - path to NYU dataset .mat file
            topk_labels - (list) top k classes for segmentation
            transform - transforms for segmentation labels and depth maps
            normalize - determines whether to apply ImageNet normalization to RGB images
            depth_norm - Normalization for depth map based on training data
            extra_augs - determines whether to apply extra agumentations to RGB images
        """
        self.topk_labels = topk_labels
        self.transform = transform

        # image net normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std) if normalize else None

        # arbitrary normalization factor for depth based on train data
        self.depth_norm = depth_norm

        # open .mat file as an h5 object
        self.h5_obj = h5py.File(root, mode='r')

        # obtain desired groups
        self.images = self.h5_obj['images']  # rgb images
        self.depths = self.h5_obj['depths']  # depths
        self.labels = self.h5_obj['labels']  # sematic class mask for each image
        self.names = self.h5_obj['names']  # sematic class labels
        # self.instances = self.h5_obj['instances'] # instances
        # self.namesToIds = self.h5_obj['namesToIds']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.rotate(self.images[idx].transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE)  # rgb image
        depth = cv2.rotate(self.depths[idx], cv2.ROTATE_90_CLOCKWISE)  # depth map
        label = cv2.rotate(self.labels[idx], cv2.ROTATE_90_CLOCKWISE).astype(np.float32)  # semantic segmentation label

        # reduce to topk labels (by placing them in the uncategorized class)
        if self.topk_labels:
            for lbl in np.unique(label).astype(int):
                if lbl not in self.topk_labels:
                    label[label == lbl] = 0

        if self.transform:
            """ We need to apply the same random transforms to the image and mask,
                typically we would just place everything in a dict and use custom
                transform classes. However, we have a limited amount of training 
                data and will likely want to add aggressive augmentation. Instead
                we will get current random state before first transform and update 
                the state before each subsequent transform.
            """
            state = torch.get_rng_state()
            image = self.transform(image)

            # reset random state to that of the previous transform
            torch.set_rng_state(state)
            depth = self.transform(depth)

            # reset random state to that of the previous transform
            torch.set_rng_state(state)
            label = self.transform(label)

        # apply normalizations
        if self.normalize:
            image = self.normalize(image)
            depth = depth / self.depth_norm

        return image, (depth, label)

    def str_label(self, idx):
        """
            Obtains string label for a class index. Names/Labels are indexed from 1,
            this function is able to take this into account by subtracting 1.
            In the NYU depth dataset, labels equal 0 are considered unlabeled.
        """
        if idx - 1 < 0:
            return 'unlabeled'
        return ''.join(chr(i[0]) for i in self.h5_obj[self.names[0, idx - 1]])

    def close(self):
        self.h5_obj.close()

    def __exit__(self, *args):
        self.close()


# %% NYU dataset class from the ra parsed data

class NyuDataset_raw(Dataset):
    def __init__(self, root, topk_labels=None, transform=None, normalize=False, depth_norm=10):
        """
            NYU Depth Dataset
            root - path to NYU dataset .mat file
            topk_labels - (list) top k classes for segmentation
            transform - transforms for segmentation labels and depth maps
            normalize - determines whether to apply ImageNet normalization to RGB images
            depth_norm - Normalization for depth map based on training data
            extra_augs - determines whether to apply extra agumentations to RGB images
        """
        self.topk_labels = topk_labels
        self.transform = transform

        # image net normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std) if normalize else None

        # arbitrary normalization factor for depth based on train data
        self.depth_norm = depth_norm

        # open .mat file as an h5 object
        self.h5_obj = h5py.File(root, mode='r')

        # obtain desired groups
        self.images = self.h5_obj['images']  # rgb images
        self.depths = self.h5_obj['depths']  # depths
        self.labels = self.h5_obj['labels']  # sematic class mask for each image
        self.names = self.h5_obj['names']  # sematic class labels
        # self.instances = self.h5_obj['instances'] # instances
        # self.namesToIds = self.h5_obj['namesToIds']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.rotate(self.images[idx].transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE)  # rgb image
        depth = cv2.rotate(self.depths[idx], cv2.ROTATE_90_CLOCKWISE)  # depth map
        label = cv2.rotate(self.labels[idx], cv2.ROTATE_90_CLOCKWISE).astype(np.float32)  # semantic segmentation label

        # reduce to topk labels (by placing them in the uncategorized class)
        if self.topk_labels:
            for lbl in np.unique(label).astype(int):
                if lbl not in self.topk_labels:
                    label[label == lbl] = 0

        if self.transform:
            """ We need to apply the same random transforms to the image and mask,
                typically we would just place everything in a dict and use custom
                transform classes. However, we have a limited amount of training 
                data and will likely want to add aggressive augmentation. Instead
                we will get current random state before first transform and update 
                the state before each subsequent transform.
            """
            state = torch.get_rng_state()
            image = self.transform(image)

            # reset random state to that of the previous transform
            torch.set_rng_state(state)
            depth = self.transform(depth)

            # reset random state to that of the previous transform
            torch.set_rng_state(state)
            label = self.transform(label)

        # apply normalizations
        if self.normalize:
            image = self.normalize(image)
            depth = depth / self.depth_norm

        return image, (depth, label)

    def str_label(self, idx):
        """
            Obtains string label for a class index. Names/Labels are indexed from 1,
            this function is able to take this into account by subtracting 1.
            In the NYU depth dataset, labels equal 0 are considered unlabeled.
        """
        if idx - 1 < 0:
            return 'unlabeled'
        return ''.join(chr(i[0]) for i in self.h5_obj[self.names[0, idx - 1]])

    def close(self):
        self.h5_obj.close()

    def __exit__(self, *args):
        self.close()


# %% dataset for waterType simulation by UWCNN

class SimulationWaterTypeDataset(Dataset):

    def __init__(self, root_dir, water_type, transform=None):
        self.root_dir = root_dir

        # hard coded
        self.images_number = [9, 14, 31, 84, 85, 86, 89, 922, 96, 108, 118, 224, 226, 229, 238, 270, 278, 284, 292, 296,
                              303, 310, 322, 329, 349, 352, 385, 398, 419, 433, 436, 459, 485, 490, 551, 601, 603, 613,
                              624, 632, 635, 641, 699, 725, 745, 746, 757, 760, 764, 768, 784, 791, 802, 810, 811, 822,
                              853, 866, 875, 880, 885, 891, 901, 906, 914, 953, 965, 993, 1015, 1303, 1045, 1049, 1053,
                              1181, 1194, 1204, 1206, 1219, 1224, 1246, 1255, 1256, 1292, 1333, 1344, 1350, 1353, 1364,
                              1366, 1368, 1374, 1397, 1406, 1409, 1416, 1419, 1426, 1434, 1443, 1449]

        # self.images_number = [1181, 1026, 1416]

        uwcnn_dataset_path_type = pjoin(root_dir, f"type{water_type}_data")
        uw_path = pjoin(uwcnn_dataset_path_type, f"underwater_type_{water_type}")
        uw_paths_tmp = glob.glob(pjoin(uw_path, '*.bmp'))
        self.images_list = natsorted([path_ii for path_ii in uw_paths_tmp if
                                      int(os.path.basename(path_ii).split("_")[0]) in self.images_number])

        gt_path = pjoin(uwcnn_dataset_path_type, f"gt_type_type_{water_type}")
        gt_paths_tmp = glob.glob(pjoin(gt_path, '*.bmp'))
        self.gt_list = natsorted([path_ii for path_ii in gt_paths_tmp if
                                  int(os.path.basename(path_ii).split("_")[0]) in self.images_number])

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        gt_image = Image.open(self.gt_list[idx])

        if self.transform is not None:
            image = self.transform(image)
            gt_image = self.transform(gt_image)

        return [image, gt_image], os.path.basename(self.images_list[idx])
