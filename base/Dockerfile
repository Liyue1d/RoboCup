FROM debian:8.1

MAINTAINER Thomas Henn

# Install dependencies -------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y wget build-essential dh-autoreconf \
libboost-dev zlib1g-dev zlib1g flex libboost-filesystem-dev libboost-system-dev \
python3 python3-dev python3-pip python3-gdbm screen git gfortran libopenblas-dev liblapack-dev \
libfreetype6-dev pkg-config
RUN apt-get install -y libqtcore4 libqt4-dev libaudio-dev libpng-dev libxi-dev libglib2.0-dev libfontconfig-dev libxrender-dev libxt-dev libsm-dev libice-dev
# ----------------------------------------------------------------------------------------------------------

# General installation directory
RUN mkdir /robocup/

# rcssserver installation ----------------------------------------------------------------------------------
ADD payload/base/libbison-dev_2.7.1.dfsg-1_amd64.deb /robocup/
ADD payload/base/bison_2.7.1.dfsg-1_amd64.deb /robocup/
RUN dpkg -i /robocup/libbison-dev_2.7.1.dfsg-1_amd64.deb && dpkg -i /robocup/bison_2.7.1.dfsg-1_amd64.deb \
    && rm /robocup/libbison-dev_2.7.1.dfsg-1_amd64.deb && rm /robocup/bison_2.7.1.dfsg-1_amd64.deb

ADD payload/base/rcssserver-15.3.0_compiled.tar.gz /robocup/
RUN cd /robocup/rcssserver-15.3.0/ && make install
RUN echo "/usr/local/share" >> /etc/ld.so.conf && echo "/usr/local/lib" >> /etc/ld.so.conf && ldconfig
# ---------------------------------------------------------------------------------------------------------

# Soccerwindow installation -------------------------------------------------------------------------------
ADD payload/base/rctools-librcsc.tar.gz /robocup/
ADD payload/base/rctools-soccerwindow2.tar.gz /robocup/
RUN cd /robocup/rctools-soccerwindow2/ && make install && echo "/robocup/rctools-librcsc/lib/" >> /etc/ld.so.conf && ldconfig
# ---------------------------------------------------------------------------------------------------------

# Teams setup ----------------------------------------------------------------------------------------------

#Version from 31/07/2015
# ADD payload/base/rctools-agent2d.tar.gz /robocup/

# ----------------------------------------------------------------------------------------------------------
