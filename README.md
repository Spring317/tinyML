# TinyML with MCUNet

This is a fork of the MCUNet repository for utilizing TinyML to accelerate ML inferencing on microcontrollers and resource-constrained devices.

## Overview

MCUNet is a framework designed to bring deep learning to IoT devices with extremely limited computing resources. This project extends MCUNet with additional tools for training, evaluation, and deployment of models for specific applications.

## Getting Started

### Prerequisites

* Python 3.8+ 
* CUDA-compatible GPU (optional, for faster training)
* At least 8GB RAM
* git
* Virtual Environment (Conda or Python)

### Installation

* Note: This implementation using conda virtual environment. You could also use the python virtual environment. 

1. Clone this repository:
```bash
git clone https://github.com/Spring317/tinyML.git
cd tinyML
```

2. Create and activate a conda environment:
```bash
conda create -n tinyml 
conda activate tinyml
pip install -r requirements.txt
```

* Note: This implementation will deploy the model to raspberry pi 4 using ONNX framework. If you wish to deploy the model on the micro-controler, please refer to [tinyengine](https://github.com/mit-han-lab/tinyengine) repo

### Training and validating:

1. Creating dataset:
```bash
# Please refer to this repo https://github.com/visipedia/inat_comp/tree/master/2017 if you having problem with the dataset
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz

tar -xvzf train_val_images.tar.gz
```  

2. Modify the config.yaml file to point to your data directory:
```yaml

global:
   # In this experiment, we only use the Insecta class from the iNaturalist dataset
   # Add more superclasses if you wish to use more classes
   included_classes: ["Insecta"]
   verbose: false
   overwrite: true

#Modify these paths to the dataset you want to use
paths:
   #path for the dataset you just downloaded
   src_dataset: "/home/quydx/mcunet_full/inat_2017"
   #path for the sub-dataset you want to do training and validation (e.g. haute_garonne)
   dst_dataset: "/home/quydx/iNaturelist_transfer_learning_pytorch/data/haute_garonne"

   #I am too lazy to change the path in the code, so I just put it here. Please don't touch it T.T
   web_crawl_output_json: "/home/quydx/iNaturelist_transfer_learning_pytorch/output/haute_garonne.json"
   output_dir: "/home/quydx/iNaturelist_transfer_learning_pytorch/output"

#In case you want to have more updated dataset, you can use the web_crawl to download the latest data from iNaturalist
#Else don't touch this part
web_crawl:
total_pages: 104
base_url: "https://www.inaturalist.org/check_lists/32961-Haute-Garonne-Check-List?page="
delay_between_requests: 1.0

#Modify these parameters to your needs
train_val_split:
train_size: 0.8
random_state: 42
dominant_threshold: 0.9
```

3.  Great! Now you could start the training with very *Easy* command:
```bash
   # Default settings
   python train.py

   # Custom settings
   python train.py --epochs 100 --model mcunet-in4 --threshold 0.7

   # With custom image size
   python train.py --img_size 224 224 --batch_size 32

   # Full custom configuration
   python train.py --epochs 150 --model mcunet-in2 --threshold 0.6 \
                  --batch_size 32 --lr 0.0005 --workers 4 \
                  --img_size 192 192 --output_dir models/custom

```

4. Afterward you could make some *easy* validation based on the val.py script
```bash
   python3 eval.py
```

5.  You are good to go! Now all you have to do is to quantize the model and deploy it on the Pi
```bash
   python3 quantize.py
```

6.  For Deploying into Raspberry Pi 4b, please refer to the deployment repo [here]()



