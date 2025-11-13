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




# Version history

