```
 ██████╗ ██████╗      ██╗███████╗ ██████╗████████╗██╗██╗   ██╗███████╗███████╗
██╔═══██╗██╔══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██║██║   ██║██╔════╝██╔════╝
██║   ██║██████╔╝     ██║█████╗  ██║        ██║   ██║██║   ██║█████╗  ███████╗
██║   ██║██╔══██╗██   ██║██╔══╝  ██║        ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║
╚██████╔╝██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║ ╚████╔╝ ███████╗███████║
 ╚═════╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝
```
![build docs](https://github.com/robmacc/objectives/workflows/build%20docs/badge.svg)
![linting](https://github.com/robmacc/objectives/workflows/linting/badge.svg)
![testing](https://github.com/robmacc/objectives/workflows/testing/badge.svg)
[![codecov](https://codecov.io/gh/robmacc/objectives/branch/master/graph/badge.svg)](https://codecov.io/gh/robmacc/objectives)

## Environment

### Windows
Install CUDA:
```
choco install cuda --version 10.1
```
Install conda:
```
choco install anaconda3
```
Install conda dependencies and pip:
```
conda env create -f environment.yml
conda activate objectives
```
Install PyTorch requirements:
```
bash install-torch.sh
```

### Linux
Install CUDA:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```
Install conda:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
Install conda dependencies and pip:
```
conda env create -f environment.yml
conda activate objectives
```
Install PyTorch requirements:
```
bash install-torch.sh
```

<!-- Download cuDNN and extract the files to the CUDA directory: -->
<!-- ``` -->
<!-- [cuDNN-download-dir]/cuda/bin/cudnn65_7.dll > `NVIDIA GPU Computing Toolkit`/CUDA/vX.X/bin/ -->
<!-- [cuDNN-download-dir]/cuda/include/cudnn.h > `NVIDIA GPU Computing Toolkit`/CUDA/vX.X/include/ -->
<!-- [cuDNN-download-dir]/cuda/lib/x64/cudnn.lib > `NVIDIA GPU Computing Toolkit`/CUDA/vX.X/lib/x64/ -->
<!-- ``` -->

## Docs and Tests
Build docs with `make build-docs`,
run tests with `pytest`.

## Usage
run from project root with `python src/main.py`