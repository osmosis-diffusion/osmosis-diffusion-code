FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN pip install natsort
WORKDIR /home/osmosis-diffusion-code/



#docker build -t osmosis_docker .
#docker run -v %cd%:/home/osmosis-diffusion-code --gpus all -it --rm osmosis_docker
