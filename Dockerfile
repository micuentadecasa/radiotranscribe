FROM tensorflow/tensorflow:latest-gpu-jupyter


RUN apt-get update && apt-get install -y git
RUN mkdir /init


COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt 

# For using jupyter notebooks
RUN pip3 install -U ipykernel

# put later in requirements - here for rebuilding the devcontainer
RUN pip3 install torch

# whisper reqs
RUN apt update && apt install ffmpeg -y
RUN pip3 install git+https://github.com/openai/whisper.git 

 
#RUN python3 -m pip install ipykernel -U --user --force-reinstall