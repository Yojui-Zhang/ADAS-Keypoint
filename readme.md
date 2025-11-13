**切換TensorRT/TFlite引擎**
- cmake .. -DENGINE=TFLITE
- cmake .. -DENGINE=TENSORRT

**Ubuntu 須安裝gl 套件**
- sudo apt-get install libglfw3 libglfw3-dev

**Ubuntu 須安裝Open CL 套件**
- sudo apt install ocl-icd-opencl-dev

**Ubuntu 須安裝CUDA 套件**
- echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
- echo 'export CUDACXX=/usr/local/cuda/bin/nvcc' >> ~/.bashrc
- source ~/.bashrc

**======================================================================**
# Environment
```

 - RECOMMENDED CUDA >= 11.4
 - RECOMMENDED TensorRT >= 8.4
 - OpenCV
 
 - Single Class:
   * OpenCV version not required
   
 - Mulit Class:
   * OpenCV version >= 4.7
   
```
---------------------------------------------
# Transform Engine(Example)
```
trtexec --onnx=a.onnx --saveEngine=a.engine --fp16
```
---------------------------------------------
# Env Export

1. **Open Bashrc**
```
sudo gedit ~/.bashrc

```

2. **Add Env In Bashrc**
```
(Add TensorRT, The actual path is modified according to your own environment...)
export PATH=/usr/src/tensorrt/bin:$PATH

(Add OpenCL, The actual path is modified according to your own environment...)
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/usr/include/CL:$C_INCLUDE_PATH
```
---------------------------------------------
# install Environment

## Common Env
```
sudo apt install gcc
sudo apt install python3
sudo apt install cmake
sudo apt install make
sudo apt install libopencv-dev
sudo apt install python-opencv

sudo apt-get update
sudo apt-get install ocl-icd-opencl-dev opencl-headers

sudo apt-get install nvidia-opencl-dev nvidia-opencl-icd

sudo apt install clinfo

sudo apt-get install libjsoncpp-dev
sudo apt install curl

sudo apt-get install libeigen3-dev
sudo apt-get install libcurl4-openssl-dev

sudo apt install -y libcurl4-openssl-dev
```

## Install OpenCV(version not required):
- sudo apt install libopencv-dev

## Install OpenCV(version Custom):

1. ***安裝必要依賴***
```
sudo apt update
sudo apt install -y cmake git build-essential libgtk2.0-dev \
    pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev
    
sudo apt install -y libgtk-3-dev
sudo apt install -y libgtk2.0-dev pkg-config
```
2. ***下載 OpenCV 原始碼（包含 contrib)***
```
- Method 1
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```
- Method 2 (recommend)
```
download source code
```
3. ***建立編譯資料夾***
```
cd ~/opencv
mkdir build && cd build
```
4. ***cmake (可設定編譯參數 如.CUDA + GStreamer + contrib)***

- **Method 1**
```
cmake ..
```
- **Method 2**
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.2 \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D WITH_CUDNN=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D BUILD_EXAMPLES=OFF ..
    
cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=8.7 \
  -D OPENCV_DNN_CUDA=ON \
  -D BUILD_opencv_cudacodec=OFF \
  -D WITH_CUDNN=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_TBB=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules
  
cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=8.7 \
  -D OPENCV_DNN_CUDA=ON \
  -D BUILD_opencv_cudacodec=OFF \
  -D WITH_CUDNN=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_TBB=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D WITH_GTK=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules
```
5. ***編譯***
```
make -j$(nproc)
```
6. ***安裝***
```
sudo make install
sudo ldconfig
```
7. ***驗證***
```
pkg-config --modversion opencv4

python3 -c "import cv2; print(cv2.__version__)"

```
**======================================================================**


# Version history

