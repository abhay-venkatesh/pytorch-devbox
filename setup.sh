# Create a conda environment
conda create --name pytorch python=3.6
source activate pytorch

# Install PyTorch and TorchVision
conda install pytorch torchvision -c pytorch

# Install other packages
conda install --file requirements.txt

# Install COCO API
conda install -c hcc pycocotools
