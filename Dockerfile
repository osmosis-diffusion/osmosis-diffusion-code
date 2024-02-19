FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN pip install natsort
WORKDIR /home/osmosis-diffusion-code/
