FROM balenalib/raspberrypi5-debian:bookworm-build

WORKDIR /root

# Install generic requirements
RUN apt-get update && \
    apt-get install -y software-properties-common wget

# Need to create a sources.list file for apt-add-repository to work correctly:
# https://groups.google.com/g/linux.debian.bugs.dist/c/6gM_eBs4LgE
RUN echo "# See sources.lists.d directory" > /etc/apt/sources.list

# Add Raspberry Pi repository, as this is where we will get the Hailo deb packages
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
    apt-add-repository -y -S deb http://archive.raspberrypi.com/debian/ bookworm main

# Fake systemd so hailoRT will install in container:
RUN echo '#!/bin/sh\nexec "$@"' > /usr/bin/sudo && chmod +x /usr/bin/sudo
RUN echo '#!/bin/bash\nexit 0' > /usr/bin/systemctl && chmod +x /usr/bin/systemctl
RUN mkdir -p /run/systemd && echo 'docker' > /run/systemd/container

# Dependencies for hailo-tappas-core
RUN apt-get install -y python3 ffmpeg x11-utils python3-dev python3-pip python3-venv \
    python3-setuptools gcc-12 g++-12 python-gi-dev pkg-config libcairo2-dev \
    libgirepository1.0-dev libgstreamer1.0-dev cmake \
    libgstreamer-plugins-base1.0-dev libzmq3-dev rsync git \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-libcamera libopencv-dev \
    python3-opencv python3-picamera2

RUN mkdir -p hailort
RUN git clone --branch master-v4.18.1 --depth 1 https://github.com/hailo-ai/hailort.git hailort/sources

RUN cd hailort/sources && cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DHAILO_BUILD_EXAMPLES=1 && sudo cmake --build build --config release --target install
    

WORKDIR /app
# create python virtual env
RUN python3 -m venv venv --system-site-packages
RUN . venv/bin/activate && pip3 install -U vidgear[asyncio] uvicorn
COPY . . 

CMD ["balena-idle"]