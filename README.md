# Osmosis: RGBD Diffusion Prior for Underwater Image Restoration

### [Paper](link) , [Project Page](https://osmosis-diffusion.github.io/)

> Osmosis: RGBD Diffusion Prior for Underwater Image Restoration
>
> [Opher Bar Nathan](mailto:barnathanopher@gmail.com) | Deborah Levy | [Tali Treibitz](https://www.viseaon.haifa.ac.il/) | [Dan Rosenbaum ](https://danrsm.github.io/)

This repository contains official PyTorch implementation for **Osmosis: RGBD Diffusion Prior for Underwater Image Restoration**.

![intro](figures/teaser2.png)

This code is based on [guided-diffusion](https://github.com/openai/guided-diffusion), [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior)


## Abstract

In this work, our proposed method takes an underwater-captured image as input and outputs a restored image, free from the distorting effects of the water medium, along with a depth estimation of the scene.
Notably, our method operates with a single image as input.
We show how to leverage in-air images to train diffusion prior for underwater restoration. We observe that only color data is insufficient, and augment the prior with a depth channel. 
Thus, we train an unconditional diffusion model prior on the joint space of color and depth, using standard RGBD datasets of natural outdoor scenes in air.
Using this prior together with a novel guidance method based on the underwater image formation model, we generate posterior samples of clean images, effectively eliminating the water effects. 
 Despite the trained prior not being exposed to underwater images, our method successfully performs image restoration and depth estimation on underwater scenes.

## RGBD Prior

In the course of this research, an unconditional Diffusion Probabilistic Model (DDPM) is trained on RGBD (color image and depth map) data. The training follows [improved-diffusion](https://github.com/openai/guided-diffusion) and [guided-diffusion](https://github.com/openai/guided-diffusion).
To adapt the model for RGBD data (instead of RGB), we made specific modifications by adjusting the UNet input layer to handle 4 channels and the output layers to generate 8 channels.

<!--We initialize the model by a pre-trained unconditional DDPM on ImageNet, provided by [guided-diffusion](https://github.com/openai/guided-diffusion)-->

The new prior is trained using 4 outdoor RGBD datasets: [DIODE](https://diode-dataset.org/) (only outdoor scenes), [HRWSI](https://github.com/KexianHust/Structure-Guided-Ranking-Loss?tab=readme-ov-file), [KITTI](https://www.cvlibs.net/datasets/kitti/) and [ReDWeb-S](https://github.com/nnizhang/SMAC).

The trained RGBD prior, named "osmosis_outdoor.pt," can be downloaded from the provided [link](https://drive.google.com/file/d/13o2roYPI-2wDOh8LvPHGQIrqRommSuJT/view?usp=drive_link)

## Datasets

The method is specifically designed for underwater scenes. 

Consequently, underwater images are supplied, and simulated data was also examined for quantitative analysis. 

Furthermore, the algorithm exhibits versatility for additional tasks such as dehazing, hence, a set of images with haze is included.



<br />

### Underwater images - real data - [link](https://drive.google.com/drive/folders/1mlojrmSsSF07y5jH3m1P7SBlY5TF0C7A?usp=sharing)

This folder contains two similar datasets.
- **Low** Resolusion set - [link](https://drive.google.com/drive/folders/1g6WAF6RAQlen84bMFNIMq-U-3XJ7oN65?usp=sharing) - 256x256
- **High** Resolusion set - [link](https://drive.google.com/drive/folders/12c8MDPEHgOSSMLZ0l-eFCs8iIQoOPuVN?usp=sharing)

Both datasets contain identical images, with the *Low-Resolution Set* serving as a cropped and resized version of the *High-Resolution Set* images.

Our method accepts input images of any resolution, but it standardizes the resolution by resizing them to 256 pixels on the smaller side and subsequently center cropping them.

The underwater images are sourced from three datasets: [SQUID](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html), 
[SeaThru](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html) 
and [SeaThru-Nerf](https://sea-thru-nerf.github.io/).

The images are linear (were not undergo any non-linear processing) and undergo a white balance process.

<br />

### Underwater images - Simulated data with Ground Truth

As part of this study, underwater scenes were simulated to facilitate quantitative comparisons. 

A set of 100 images was sourced from the indoor dataset [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html),
each accompanied by its corresponding depth map. 

Two datasets are involved in this simulation. 

The specific simulation parameters are outlined in both the paper and the supplementary data.

- [Simulation #1](https://drive.google.com/drive/folders/1E4cXHtpNWX3wHrkmiVDF_XUpPnWMOe-0?usp=sharing)
- [Simulation #2](https://drive.google.com/drive/folders/1_096PIqXR0w4j8ASyZsEPHs1WZj3AscN?usp=sharing)

Each simulation includes 3 folders: 
1. input - the simulated images
2. gt_rgb - Ground Truth color images
3. gt_rgb - Ground Truth depth maps

<br />

### Hazed images - [link](https://drive.google.com/drive/folders/18Xpy8MdsIucNIRhTTKD_Q3isbC79TW89?usp=sharing)

We present preliminary results of applying this method to the dehazing task, therefore, we provide several images captured in hazed conditions.

<br />

### Using your own data

In case you would like to try this method on your own data:
- Place all images in the same folder.
- In the configurations file, modify the field ``` data: root: <path> ``` to the folder path.
- Specify the name in the ``` data: name: <dataset_name> ``` field; the results will be saved into a folder with the same name.
- If there is ground truth data, indicate its path in the ``` data: gt_rgb: <path> ``` and ``` data: gt_depth: <path> ``` fields. Change the flag ``` data: ground_truth: True ``` (similar to the configurations in ```osmosis_simulation_sample_config.yaml```).
- If your data is not simulated or is not include linear images, set the flag ``` degamma_input: True ```, as it often produces improved results.

<br />

## Prerequisites

See the environment file: [link](environment.yml)


<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/osmosis-diffusion/osmosis-diffusion-code

cd osmosis-diffusion-code
```

<br />

### 2) Download pretrained checkpoint and data

### Checkpoint

Create a new dirctory ```./models```.

From the [link](https://drive.google.com/file/d/13o2roYPI-2wDOh8LvPHGQIrqRommSuJT/view?usp=drive_link), download the checkpoint "osmosis_outdoor.pt" into models directory.

<br />

### Datasets

Create a new dirctory ```./data```.

Download the relevant [dataset](#datasets) into ```./data/``` directory.


<br />

### 3) Set environment


###  Local environment setting

Install dependencies

```
conda create -n osmosis python=3.8

conda activate osmosis
```

See dependencies at environment.yml file - [link](environment.yml)

<!--
### [Option 2] Build Docker image

Install docker engine, GPU driver and proper cuda before running the following commands.

Dockerfile already contains command to clone external codes. You don't have to clone them again.

--gpus=all is required to use local GPU device (Docker >= 19.03)

```
docker build -t dps-docker:latest .

docker run -it --rm --gpus=all dps-docker
```

-->


<br />

### 4) Inference

The configuration file structure is thoroughly outlined in this [section](#structure-of-configurations-file),
enabling users to modify configurations and fine-tune parameters for experimental purposes.

By default, results are saved in the directory ```./results/<task name>/<dataset name>/<date>/<run#>```.

Additionally, both a log file and configuration file are stored in the same path.

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


#### e) Sample from RGBD Prior - **Without** guidance

In this scenario, there is no guidance provided for the sampling process, resulting in the production of an RGB image and its corresponding depth map. 

The absence of guidance implies no constraints on achieving a visually coherent image.

```
python RGBD_prior_sampling.py --config_file ./configs/RGBD_sample_config.yaml
```

<br />


## Structure of configurations file

In this section the structure and the *relevant* fields in the configuration file are explained.


```
save_dir: results    # saving directory path - it will be saved under the running directory

degamma_input: False # should be True in case of NOT linear images, or NOT simulated images, otherwise False
manual_seed: 0       # manuual seed for the diffusion sampling process
check_prior: False   # relevant only for the check prior inference

save_singles: True   # save single results images - 1)reference image (input), 2)restored RGB image and 3)depth estimation image
save_grids: True     # save grid of the results, next to each other

record_process: True # record the sampling process
record_every: 200    # in case "record_process: True" - record every <value> steps (in this case - 200)

# change unet input and output - for RGBD - it is
change_input_output_channels: True
input_channels: 4   # RGBD
output_channels: 8  # RGBD * 2 - learning sigma = True, if False 4

sample_pattern:     # the diffusion sampling pattern for the 
  pattern: pcgs     # original, pcgs - from gibbsDDRM

  # relevant only for "pattern: pcgs"
  # update phi's
  update_start: 0.7    # optimizing phi's (<value>*T)
  update_end: 0        
  global_N: 1          # repeat several times the T steps
  local_M: 1           # number of iterations between update x_t and optimizing phis for the same t - time step
  n_iter: 20           # for each t step, the number of optimization steps for te phi's
  
  start_guidance: 1    # PGDiff - when to guide? no guidance at all not in the range (<value>*T)
  stop_guidance: 0


unet_model:                      # unet model configurations
  model_path: osmosis_outdoor.pt # pretrained model file name (should be in the ./models/ directory)
  pretrain_model: osmosis        # pretrained model name

conditioning:
  method: osmosis                       # conditioning method - osmosis, ps 

params:    
    loss_weight: depth                 # none, depth # if "none" so the rest has no meaning
    weight_function: gamma,1.4,1.4,1   # function,original- [0,1], gamma=((x+value[0])*value[1])^value[2]
    scale: 7,7,7,0.9                   # guidance scale for each channel (RGBD)
    gradient_clip: True,0.005          # gradient clipping value (is True)

# specify the loss and its weight/scale, if not specified so no auxiliary loss
# see the paper for details on the losses
aux_loss:
  aux_loss:
    avrg_loss: 0.5        # scale of that loss
    val_loss: 20          # scale of that loss

data:
  name: osmosis                      # dataset name
  root: .\data\underwater\high_res   # path of the dataset
  ground_truth: False                # if the dataset includes ground truth - True, else - False
  gt_rgb: .\data\simulation_1\gt_rgb        # dataset ground truth paths - comment when no GT data
  gt_depth: .\data\simulation_1\gt_depth    # dataset ground truth paths - comment when no GT data


measurement:
  operator:

    name: underwater_physical_revised # underwater_physical_revised, haze_physical, noise (for check prior)
    optimizer: sgd                    # water parameters optimizer - options are adam, sgd

    depth_type: gamma                 # original- [0,1], gamma=((x+value[0])*value[1])^value[2]
    value: 1.4,1.4,1

    phi_a: 1.1,0.95,0.95              # initalized values
    phi_a_eta: 1e-5                   # step size for the optimization
    phi_a_learn_flag: True            # optimization flaf - if False, there is no optimization for this parameter  

    phi_b: 0.95, 0.8, 0.8             # initalized values
    phi_b_eta: 1e-5                   # step size for the optimization
    phi_b_learn_flag: True            # optimization flaf - if False, there is no optimization for this parameter  

    phi_inf: 0.14, 0.29, 0.49         # initalized values
    phi_inf_eta: 1e-5                 # step size for the optimization
    phi_inf_learn_flag: True          # optimization flaf - if False, there is no optimization for this parameter  

  noise:                              # added noise
    name: clean                       # clean - osmosis, gaussian - ps
    sigma: 0.001                      # comment in case of "clean" uncomment in case of "gaussian"

```

<!--
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
-->
