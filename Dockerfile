FROM jupyter/minimal-notebook:notebook-6.4.3

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt 

# For using jupyter notebooks
RUN pip3 install -U ipykernel

# put later in requirements - here for rebuilding the devcontainer
RUN pip3 install torch

 
#RUN python3 -m pip install ipykernel -U --user --force-reinstall