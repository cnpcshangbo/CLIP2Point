# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CC=gcc-11
ENV CXX=g++-11
ENV TORCH_CXX_FLAGS="-std=c++17"
ENV NVCC_FLAGS="-std=c++17"
ENV CXXFLAGS="-std=c++17"
ENV CFLAGS="-std=c++17"

# Install system dependencies and gcc-11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
    curl \
    tmux \
    libblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    cmake \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    libpthread-stubs0-dev \
    gcc-11 \
    g++-11 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-11 and g++-11 as the default compilers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-11 100 \
    && update-alternatives --set gcc /usr/bin/gcc-11 \
    && update-alternatives --set g++ /usr/bin/g++-11 \
    && update-alternatives --set cc /usr/bin/gcc-11 \
    && update-alternatives --set c++ /usr/bin/g++-11

# Verify all compiler versions point to gcc-11
RUN echo "Verifying compiler versions:" && \
    gcc --version && \
    g++ --version && \
    cc --version && \
    c++ --version && \
    echo "#include <iostream>\nint main() { if constexpr(true) { return 0; } return 1; }" > test.cpp && \
    g++ -std=c++17 test.cpp -o test && \
    rm test.cpp test

# Verify CUDA installation and accessibility
# Note: GPU availability checks (nvidia-smi) should be performed at runtime
# using: docker run --gpus all ...
RUN echo "CUDA_HOME: $CUDA_HOME" && \
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && \
    echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES" && \
    echo "NVIDIA_DRIVER_CAPABILITIES: $NVIDIA_DRIVER_CAPABILITIES" && \
    echo "CC: $CC" && \
    echo "CXX: $CXX" && \
    echo "TORCH_CXX_FLAGS: $TORCH_CXX_FLAGS" && \
    ls -l $CUDA_HOME && \
    nvcc --version && \
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'None'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Install PyTorch3D dependencies
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# Update conda
RUN conda update -n base -c defaults conda -y

# Install and configure CUB
WORKDIR /tmp
RUN rm -rf /tmp/cub && \
    git clone https://github.com/NVIDIA/cub.git /tmp/cub && \
    export CUB_HOME=/tmp/cub

# # Build PyTorch3D from source with C++17 flags
RUN git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd pytorch3d && git checkout v0.7.7 && \
    FORCE_CUDA=1 \
    PYTORCH3D_NO_NINJA=1 \
    MAX_JOBS=4 \
    CUB_HOME=/tmp/cub \
    TORCH_CUDA_ARCH_LIST="8.6" \
    TORCH_CXX_FLAGS="-std=c++17" \
    NVCC_FLAGS="-std=c++17" \
    CXXFLAGS="-std=c++17" \
    CFLAGS="-std=c++17" \
    CC=gcc-11 \
    CXX=g++-11 \
    pip install -e .

# Test if PyTorch3D import works
RUN python -c "from pytorch3d.structures import Meshes; print('PyTorch3D import successful!')"

# Install CLIP
RUN pip install ftfy regex tqdm
RUN apt-get update --allow-insecure-repositories && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/openai/CLIP.git

# Install pointnet2_ops and other requirements
WORKDIR /tmp/pointnet2

RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git . && \
    # cd Pointnet2_PyTorch && \
    git fetch origin pull/186/head:pr-186 && \
    git checkout pr-186


# RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git .
ENV CUDA_HOME=/usr/local/cuda
RUN cd pointnet2_ops_lib && \
    TORCH_CUDA_ARCH_LIST="8.6" \
    TORCH_CXX_FLAGS="-std=c++17" \
    NVCC_FLAGS="-std=c++17" \
    CXXFLAGS="-std=c++17" \
    CFLAGS="-std=c++17" \
    CC=gcc-11 \
    CXX=g++-11 \
    python setup.py build_ext --inplace && \
    python setup.py install
WORKDIR /app

# Install packages one by one to identify CUDA 12 dependency
RUN pip install plyfile h5py && \
    pip install "typing-extensions<4.4.0" "fsspec[http]<2022.5.0" "torchmetrics<0.7.0" && \
    pip install "lightly==1.2.0" && \
    pip install torch_optimizer && \
    pip install open3d

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p datasets data/ModelNet40_Align data/ModelNet40_Ply data/Rendering data/ShapeNet55

# Set default command to activate conda environment
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
