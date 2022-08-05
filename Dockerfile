FROM ros:kinetic
WORKDIR /root/
# install related pkgs
RUN apt-get update && apt-get install -y wget apt-utils software-properties-common git unzip \ 
    cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libproj-dev libpcap0.8-dev\
    build-essential libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev \
    python-pip python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    linux-libc-dev cmake cmake-gui libusb-1.0-0-dev libusb-dev libudev-dev mpi-default-dev openmpi-bin openmpi-common \
    libflann1.8 libflann-dev libeigen3-dev libboost-all-dev libvtk5.10-qt4 libvtk5.10 libvtk5-dev libqhull* libgtest-dev \
    freeglut3-dev pkg-config libxmu-dev libxi-dev mono-complete qt-sdk openjdk-8-jdk openjdk-8-jre
RUN apt-get install -y ros-kinetic-cv-bridge ros-kinetic-pcl-conversions ros-kinetic-tf
RUN pip install numpy==1.15.4
# install Ceres
RUN wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz && tar zxf ceres-solver-1.14.0.tar.gz && rm ceres-solver-1.14.0.tar.gz && \
    cd ceres-solver-1.14.0 && mkdir build && cd build && cmake .. && make -j3 && make test && make install && cd ../.. && rm -rf ceres-solver-1.14.0
# install OpenCV 3.4.3
RUN wget --no-check-certificate https://github.com/opencv/opencv/archive/3.4.3.zip && unzip 3.4.3.zip && rm 3.4.3.zip && mv opencv-3.4.3 opencv && \ 
    wget --no-check-certificate https://github.com/opencv/opencv_contrib/archive/3.4.3.zip && unzip 3.4.3.zip && rm 3.4.3.zip && mv opencv_contrib-3.4.3 opencv_contrib && \
    cd opencv && mkdir build && cd build && \ 
    cmake -D CMAKE_USE_OPENSLL=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ .. && \
    make -j3 && make install && cd ../.. && rm -rf opencv opencv_contrib
# install PCL 1.8
RUN wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.zip && unzip pcl-1.8.1.zip && rm pcl-1.8.1.zip && mv pcl-pcl-1.8.1 pcl && \
    cd pcl && mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
            -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON -DCMAKE_INSTALL_PREFIX=/usr .. && make -j5 && make install && cd ../.. && rm -rf pcl
