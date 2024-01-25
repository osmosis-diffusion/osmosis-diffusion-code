import cv2
import os
from os.path import join as pjoin
import numpy as np
import glob
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset


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

    def __init__(self, root_dir, gt_rgb_dir, gt_depth_dir, transform=None):
        self.gt_rgb_dir = gt_rgb_dir
        self.gt_depth_dir = gt_depth_dir
        self.root_dir = root_dir

        self.gt_rgb_list = natsorted(glob.glob(pjoin(gt_rgb_dir, "*.*")))
        self.gt_depth_list = natsorted(glob.glob(pjoin(gt_depth_dir, "*.*")))
        self.images_list = natsorted(glob.glob(pjoin(root_dir, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.gt_rgb_list)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.images_list[idx])
        image = Image.open(self.images_list[idx])
        gt_rgb_image = Image.open(self.gt_rgb_list[idx])

        gt_depth_image_tmp = cv2.imread(self.gt_depth_list[idx], cv2.IMREAD_UNCHANGED)
        if gt_depth_image_tmp.dtype == 'uint16':
            gt_depth_image = Image.fromarray((gt_depth_image_tmp//256).astype(np.uint8))
        else:
            gt_depth_image = Image.fromarray(gt_depth_image_tmp)
            # gt_depth_image = Image.open(self.gt_depth_list[idx])

        if self.transform is not None:
            image = self.transform(image)
            gt_rgb_image = self.transform(gt_rgb_image)

            # it is a single channel image (only depth), so preprocess is required
            # gt_depth_image = Image.merge("RGB", (gt_depth_image,gt_depth_image,gt_depth_image))
            gt_depth_image = gt_depth_image.convert(mode="RGB")
            gt_depth_image = self.transform(gt_depth_image)

        return [image, gt_rgb_image, gt_depth_image], image_name
