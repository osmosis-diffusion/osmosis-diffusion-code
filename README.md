# Osmosis: RGBD Diffusion Prior for Underwater Image Restoration

### [Paper](link) , [Project Page](https://osmosis-diffusion.github.io/)

> Osmosis: RGBD Diffusion Prior for Underwater Image Restoration
>
> [Opher Bar Nathan](mailto:barnathanopher@gmail.com) | Deborah Levy | [Tali Treibitz](https://www.viseaon.haifa.ac.il/) | [Dan Rosenbaum ](https://danrsm.github.io/)

This repository contains official PyTorch implementation for __Osmosis: RGBD Diffusion Prior for Underwater Image Restoration__.

![intro](figures/teaser2.png)

This code is based on [guided_diffusion](https://github.com/openai/guided-diffusion), [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior)


## Prerequisites

See the environment file: [link](environment.yml)

<br />


##Datasets

The method is specifically designed for underwater scenes. 

Consequently, underwater images are supplied, and simulated data was also examined for quantitative analysis. 

Furthermore, the algorithm exhibits versatility for additional tasks such as dehazing, hence, a set of images with haze is included.

<br />

### Underwater images - real data - [link](https://drive.google.com/drive/folders/1mlojrmSsSF07y5jH3m1P7SBlY5TF0C7A?usp=sharing)

This folder contains two similar datasets.
- __Low__ Resolusion set - [link](https://drive.google.com/drive/folders/1g6WAF6RAQlen84bMFNIMq-U-3XJ7oN65?usp=sharing) - 256x256
- __High__ Resolusion set - [link](https://drive.google.com/drive/folders/12c8MDPEHgOSSMLZ0l-eFCs8iIQoOPuVN?usp=sharing)

Both contain the same images, but __Low__ Resolusion set is cropped and resized vopy of the __High__ Resolusion set images.

Our methos gets as input any resolution but it is resized according the small image side to 256 pixels and then being cropped.

Underwater images are taken from 3 deataset: [SQUID](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html), 
[SeaThru](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html) 
and [SeaThru-Nerf](https://sea-thru-nerf.github.io/).

The images are raw and undergo a white balance process.

<br />

### Underwater images - Simulated data with Ground Truth

As part of this work an underwater scenes were simulated for quantitative comparisons.

100 Images were taken from the indoor dataset: [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html),
which includes images and corresponded depthmap.

There are two datasets.

The simulations parameters are detailed in the paper and in the supplementary data.

- [Simulation #1](https://drive.google.com/drive/folders/1E4cXHtpNWX3wHrkmiVDF_XUpPnWMOe-0?usp=sharing)
- [Simulation #2](https://drive.google.com/drive/folders/1_096PIqXR0w4j8ASyZsEPHs1WZj3AscN?usp=sharing)


Each simulation includes 3 folders: 
1. input - the simulated images
2. gt_rgb - Ground Truth color images
3. gt_rgb - Ground Truth depth maps

<br />

### Hazed images - [link](https://drive.google.com/drive/folders/18Xpy8MdsIucNIRhTTKD_Q3isbC79TW89?usp=sharing)

We present preliminary results of this method applied to the dehazing task.

<br />

### Using your own data

In case you would like to try this method on your own data:
- Place all images in the same folder.
- In the configurations file, modify the 'data: root: <path>' to the folder path.
- Specify the name in the 'data: name: <dataset_name>' field; the results will be saved into a folder with the same name.
- If there is ground truth data, indicate it in the 'data: gt_rgb: <path>' and 'data: gt_depth: <path>' fields. Change the flag 'data: ground_truth: True' (similar to the configurations in 'osmosis_simulation_sample_config.yaml').
- If your data is not simulated or is not include linear images, set the flag 'degamma_input: True', as it often produces improved results.

<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/DPS2022/diffusion-posterior-sampling

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint and data

### Checkpoint

Create a new dirctory "models".

From the [link](https://drive.google.com/file/d/13o2roYPI-2wDOh8LvPHGQIrqRommSuJT/view?usp=drive_link), download the checkpoint "osmosis_outdoor.pt" into models directory.

<br />

### Datasets

Create a new dirctory "data".

Download the relevant [dataset](##Datasets) into data directory.

<br />




### 3) Set environment

<!--
### [Option 1] Local environment setting

Install dependencies

```
conda create -n osmosis python=3.8

conda activate osmosis
```

See dependencies at environment.yml file



### [Option 2] Build Docker image

Install docker engine, GPU driver and proper cuda before running the following commands.

Dockerfile already contains command to clone external codes. You don't have to clone them again.

--gpus=all is required to use local GPU device (Docker >= 19.03)

```
docker build -t dps-docker:latest .

docker run -it --rm --gpus=all dps-docker
```

-->


### 4) Inference

There are 5 possible configurations:


#### a) Underwater Image Restoration and Depth Estimation - real data

Relevant for real underwater images.

```
python osmosis_sampling.py --config_file ./configs/osmosis_sample_config.yaml
```


#### b) Underwater Image Restoration and Depth Estimation - simulated data

Relevant for simulated underwater images.

```
python osmosis_sampling.py --config_file ./configs/osmosis_simulation_sample_config.yaml
```


#### c) Hazed Image Restoration and Depth Estimation

Relevant for images in haze environment.

```
python osmosis_sampling.py --config_file ./configs/osmosis_haze_sample_config.yaml
```


#### d) Check Prior

Here, guidance is exclusively based on the colored (RGB) image, with the objective of making the output RGB image closely resemble the input image. 

The depth map is subsequently estimated using prior information.

```
python osmosis_sampling.py --config_file ./configs/check_prior_sample_config.yaml
```


#### e) Sample from RGBD Prior - __Without__ guidance

In this scenario, there is no guidance provided for the sampling process, resulting in the production of an RGB image and its corresponding depth map. 

The absence of guidance implies no constraints on achieving a visually coherent image.

```
python RGBD_prior_sampling.py --config_file ./configs/RGBD_sample_config.yaml
```

<br />

### Structure of configurations file

Each inference 

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
blabla2024osmosis,
title={},
author={{,
booktitle={},
year={},
url={}
}
```
