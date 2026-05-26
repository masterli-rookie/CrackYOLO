# Environment Configuration Guide

> ⚠️ **Important: Do NOT use conda to configure the environment in AutoDL!** The CUDA in the conda environment is a stripped-down version that lacks compilation capabilities, which will cause compilation failures for libraries like Mamba. Please use the system's base environment directly.

### 1. Instance Selection
Select an **NVIDIA RTX 4090** instance in AutoDL and choose the system's pre-installed PyTorch + CUDA base image (no need to install PyTorch manually).

### 2. Preparation: Download `causal_conv1d`
- Go to [causal-conv1d v1.4.0 Releases](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0)
- Download `causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- Upload the downloaded `.whl` file to the `autodl-tmp/` directory in AutoDL.

### 3. Install Mamba Core Environment
Run the following commands sequentially:

```bash
cd autodl-tmp/
source /etc/network_turbo  # Enable network accelerator

git clone https://github.com/state-spaces/mamba.git

# Install causal_conv1d
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Verify installation
python -c "import torch; import causal_conv1d; from causal_conv1d import causal_conv1d_fn; print('causal_conv1d version:', causal_conv1d.__version__); print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Checkout Mamba version and install
cd mamba
git checkout v2.2.0
pip install transformers==4.34.0
pip install -e . --no-build-isolation

Test Mamba Environment
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print("Mamba OK, output shape:", y.shape)

Install Remaining Dependencies
# Base and compatibility downgrades
pip install pydantic==1.10.14 "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple

# Deep learning and vision core libraries
pip install ultralytics tensorboard wandb matplotlib opencv-python optuna -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "fastapi>=0.100.0" antialiased-cnns torch_dct -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchsummary lightning==1.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

# OpenMMLab series
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

# ONNX related
pip install onnx==1.14.0 onnxruntime==1.15.1 onnxsim==0.4.36 onnxruntime-gpu==1.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Other tools and data processing libraries
pip install pycocotools==2.0.7 PyYAML==6.0.1 scipy==1.13.0 easydict -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple

Custom Modules
The custom innovative modules for this project are located in the ultralytics/nn/Addmodules/ directory and can be directly referenced in the YOLO model configuration files 

Quickly  star : run  Train.py  

Training data you can contact:1971777601@qq.com


Dataset
The dataset used in this project can be downloaded from Baidu Netdisk.
Link:通过网盘分享的文件：Crack500_YOLO等4个文件
链接: https://pan.baidu.com/s/14QkZ7sqaURNycEY7Etn0LQ 提取码: stud 
--来自百度网盘超级会员v4的分享


