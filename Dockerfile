# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Set CUDA architecture flags
# ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
# ENV FORCE_CUDA=1
# Set C++ ABI and compiler flags
# ENV _GLIBCXX_USE_CXX11_ABI=1
# ENV TORCH_CXX11_ABI=1
# ENV CXXFLAGS="-std=c++17"
# ENV NVCC_FLAGS="-std=c++17"

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     curl \
#     tmux \
#     libblas-dev \
#     liblapack-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     ninja-build \
#     cmake \
#     python3-pip \
#     python3-dev \
#     git \
#     wget \
#     build-essential \
#     libpthread-stubs0-dev \
#     && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
# RUN ln -sf /usr/bin/python3 /usr/bin/python && \
#     ln -sf /usr/bin/pip3 /usr/bin/pip

# Install pip
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

# Install CUB
# RUN wget https://github.com/NVIDIA/cub/archive/refs/tags/1.17.2.tar.gz && \
#     tar xzf 1.17.2.tar.gz && \
#     mv cub-1.17.2 /usr/local/cub && \
#     rm 1.17.2.tar.gz
# ENV CUB_HOME=/usr/local/cub

# Install Mambaforge
# RUN wget "https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-Linux-x86_64.sh" -O mambaforge.sh && \
#     bash mambaforge.sh -b -p /opt/conda && \
#     rm mambaforge.sh
# ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.8
# RUN conda create -n clip2point python=3.8 -y && \
#     conda init bash && \
#     echo "conda activate clip2point" >> ~/.bashrc

# Activate environment and install packages
# SHELL ["conda", "run", "-n", "clip2point", "/bin/bash", "-c"]

# Clean any previous installations to avoid conflicts
# RUN pip uninstall -y torch torchvision torchaudio pytorch3d

# Install Intel OpenMP runtime which PyTorch needs
# RUN apt-get update && apt-get install -y libomp5 && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support using pip
# RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Print PyTorch version
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('PyTorch CUDA version:', torch.version.cuda)"

# Install CUB using apt instead of conda
# RUN apt-get update && \
#     apt-get install -y nvidia-cuda-dev && \
#     rm -rf /var/lib/apt/lists/*

# Install PyTorch3D dependencies
# RUN pip install 'fvcore>=0.1.5' && \
#     pip install 'iopath>=0.1.7' && \
#     pip install 'scikit-image>=0.17.2' && \
#     pip install 'plotly>=4.14.3' && \
#     pip install 'black==21.4b2' && \
#     pip install 'pytorch-lightning>=1.2.0' && \
#     pip install 'tensorboard>=2.4.1' && \
#     pip install 'jupyter>=1.0.0' && \
#     pip install 'matplotlib>=3.3.3' && \
#     pip install 'imageio>=2.9.0' && \
#     pip install 'imageio-ffmpeg>=0.4.3'

# Install PyTorch3D from source with latest compatible version
# RUN git clone https://github.com/facebookresearch/pytorch3d.git && \
#     cd pytorch3d && \
#     git checkout v0.7.5 && \
#     FORCE_CUDA=1 PYTORCH3D_NO_NINJA=1 MAX_JOBS=4 pip install -e .

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install pytorch3d=0.7.0 -c pytorch3d

# Test if PyTorch3D import works
RUN python -c "from pytorch3d.structures import Meshes; print('PyTorch3D import successful!')"

# Install CLIP
RUN pip install ftfy regex tqdm
RUN apt-get update --allow-insecure-repositories
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-get update
RUN apt-get install -y git build-essential
RUN pip install git+https://github.com/openai/CLIP.git

# Install pointnet2_ops and other requirements
WORKDIR /tmp/pointnet2
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git .
ENV CUDA_HOME=/usr/local/cuda
RUN cd pointnet2_ops_lib && \
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
