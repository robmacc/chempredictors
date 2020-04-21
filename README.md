```
 ██████╗ ██████╗      ██╗███████╗ ██████╗████████╗██╗██╗   ██╗███████╗███████╗
██╔═══██╗██╔══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██║██║   ██║██╔════╝██╔════╝
██║   ██║██████╔╝     ██║█████╗  ██║        ██║   ██║██║   ██║█████╗  ███████╗
██║   ██║██╔══██╗██   ██║██╔══╝  ██║        ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║
╚██████╔╝██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║ ╚████╔╝ ███████╗███████║
 ╚═════╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝
```
[![Documentation Status](https://readthedocs.org/projects/objectives/badge/?version=latest)](https://objectives.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/robmacc/objectives.svg?branch=master)](https://travis-ci.org/robmacc/objectives)
[![Coverage Status](https://coveralls.io/repos/github/robmacc/objectives/badge.svg?branch=master)](https://coveralls.io/github/robmacc/objectives?branch=master)

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
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends cuda-9-2
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