FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Update
RUN apt-get update
RUN apt update

# Install vim
RUN apt install --assume-yes vim

# Set DATAROOT
ENV DATAROOT=/

# Set tmpdir and mplconfigdir for matplotlib to /mnt/pred/
ENV TMPDIR=/mnt/pred/
ENV MPLCONFIGDIR=/mnt/pred/

# Copy files
ADD . /workspace/

# Make .sh files writable
RUN chmod +x /workspace/*.sh

RUN mkdir /mnt/data
RUN mkdir /mnt/pred

# Install requirements
RUN python3 -m pip install -r /workspace/requirements.txt

# Install uas_mood package
RUN python3 -m pip install -e .
