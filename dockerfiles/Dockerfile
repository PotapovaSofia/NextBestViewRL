FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y software-properties-common

RUN echo "deb http://ppa.launchpad.net/djcj/tools/ubuntu bionic main" >> /etc/apt/sources.list.d/djcj_tools.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4BC359BE72CD1721449C8CC24A5F4FED4B4972B

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install git-core -y

RUN apt-get install libsm6 libxext6 libxrender-dev -y
RUN apt-get install libjpeg8-dev libtiff5-dev libpng-dev -y
RUN apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
RUN apt-get install libgtk2.0-dev -y
RUN apt-get install libatlas-base-dev gfortran -y

# Socks proxy to bypass GCFW with shadowsocks.

RUN pip install trimesh
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install scikit-image
RUN pip install sklearn
RUN pip install seaborn
RUN pip install imageio
RUN pip install jupyter
RUN pip install pytorch3d
RUN pip install K3D

ENV PYTHONPATH="$PYTHONPATH:/tf"

