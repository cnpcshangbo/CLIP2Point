FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install pytorch3d=0.7.0 -c pytorch3d

# Test if PyTorch3D import works
RUN python -c "from pytorch3d.structures import Meshes; print('PyTorch3D import successful!')"