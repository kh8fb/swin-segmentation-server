# Image Attributions: The ultimate guide
#### Table of contents
1. [Installation](#1-installation)
2. [Running the Server](#2-running-the-server)
3. [Prepping the ipynotebook](#3-prepping-the-ipynotebook)
4. [Citations](#4-citations)

#### Model Paper Links
1. [Sample-Efficient Neural Architecture Search by Learning Action Space (LaNet)](https://arxiv.org/abs/1906.06832)
2. [Big Transfer (BiT):  
General Visual Representation Learning](https://arxiv.org/abs/1912.11370)
3. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) 
4. [TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/pdf/1903.12174.pdf)

# 1 Installation
## Image Classification Models

First, create an Anaconda environment:

       conda create -n image-attributions python=3.8 -y

Next, activate the environment, and `git clone` the [intermediate-gradients repo](https://github.com/kh8fb/intermediate-gradients).

     conda activate image-attributions
     git clone https://github.com/kh8fb/intermediate-gradients.git
     cd intermediate-gradients
     pip install -e .

Finally,  `git clone` the [int-gradients-image-server repo](https://github.com/kh8fb/int-gradients-image-server).  `cd` into this project's directory and install the requirements with `pip`

      git clone https://github.com/kh8fb/int-gradients-image-server.git
      cd int-gradients-image-server
      pip install -e .

## Image Segmentation Models
Both of the image segmentation model servers require the same first few initial steps for installation. Create a conda environment:

     conda create -n image-segmentation-server python=3.8 -y

Next, activate a shell window on Rivanna to ensure that you have access to the CUDA when installing cudatoolkit.  Also activate your conda environment once the allocation has occurred.

      srun --partition=bii-gpu --nodes=1 --gres=gpu:1 --time=01:00:00 -W 0 --pty $SHELL
      conda activate image-segmentation-server

Load the module for `gcccuda` and set the `$CUDA_HOME` variable

     $module load gcccuda
     $which nvcc
     /apps/software/standard/compiler/gcc/9.2.0/cuda/11.0.228/bin/nvcc 
     export CUDA_HOME=/apps/software/standard/compiler/gcc/9.2.0/cuda/11.0.228/

Finally,

	conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
	pip install opencv-python

Now that the CUDA and Pytorch has been correctly set up, follow either of the following sections depending on if you're installing TensorMask or the SwinTransformer 
    

#### TensorMask
Inside your conda environment, install the detectron2 GitHub repo 
  
      git clone https://github.com/facebookresearch/detectron2.git
      python -m pip install -e /path/to/detectron2
      pip install -e /path/to/detectron2/projects/TensorMask

And then install the server itself

    git clone https://github.com/kh8fb/tensormask-segmentation-server.git
    cd tensormask-segmentation-server/
    pip install -e .

#### Swin Transformer

For the Swin Transformer, the next step depends on which cuda and pytorch version is installed.  Following the previous steps, you should be able to do 

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.1/index.html

But the general format is 

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

Next, `pip install` opencv and clone the [Swin for Object Detection Github Repo](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

      pip install opencv-python
      git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git 
      cd Swin-Transformer-Object-Detection
      pip install -r requirements/build.txt
      pip install -v -e .

Finally, install the swin-transformer-server repo and you're all set

	 git clone https://github.com/kh8fb/swin-segmentation-server.git
	 cd swin-segmentation-server
	 pip install -e .


# 2 Running the server
First, make sure you've downloaded the model links for BiT and LaNet from [this Google Drive link](https://drive.google.com/drive/u/0/folders/1KtuVv2GPtbcuy9fifuCXySuqQhcPc-nO) and the image segmentation model links from [here](https://drive.google.com/drive/folders/1s4xvls62Z8uPAXW2jUu96Q2w1OinyEy6?usp=sharing)
Once the server's are running, they all have the same input steps, however they all have different command line arguments to get them started.  Each server should be run on host 0.0.0.0 so that you can access it from another shell window.
#### LaNet

     intgrads-images -lb /path/to/lanet_model.pth -h 0.0.0.0 -p 8008 --cuda

#### BiT

     intgrads-images -bp /path/to/bit_model.pth -h 0.0.0.0 --cuda -p 8008
#### TensorMask

      tensormask-server -tp /path/to/tensormask_model.pkl -cp /path/to/detectron2/projects/TensorMask/configs/tensormask_R_50_FPN_6x.yaml -h 0.0.0.0 -p 8008

#### Swin Transformer

      swin-server -sp /path/to/swin_model.pth -cp /path/to/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py -h 0.0.0.0 -p 8008

#### Getting Output

##### prepare_input.py
Each model accepts JSON input and thus requires some preparation before an input Image is ready for the model.  This is done with *that repo's* prepare_input.py file.  For the *classification models*, the input looks like

     python prepare_input.py /path/to/image.jpeg airplane input_json_file.json

Where the `classification` is one of ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, or ‘truck' (The classifications from the CIFAR-10 dataset the model was trained upon).  Your input will be stored in `input_file.json`.

For the *image segmentation models*, no classification needs to provided so you just need to run 

    python prepare_input.py /path/to/image.jpg input_json_file.json

##### Querying the server
Get the host's IP address using the command

    hostname -I
    10.123.45.110 10.222.222.345 10.333.345.678

The first IP address is what will be used in the `curl` command, which is the same for every server

     curl http://10.123.45.110:8008/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"


##### Understanding the output
The attributions are stored in a dictionary with the keys: "integrated_grads", "integrated_directional_grads", "step_sizes", and "intermediates".  The prediction masks from image segmentation are stored in a dictionary with the key "pred_masks". Both of these can be accessed with:

    >>> import gzip
    >>> import torch
    >>> from io import BytesIO
    >>> with gzip.open("saved_file.gzip", 'rb') as fobj:
    >>>      x = BytesIO(fobj.read())
    >>>      output_dict = torch.load(x)

# 3 Prepping the ipynotebook

Create a kernel from your `image-attributions` conda	environment by installing ipykernel and setting up 

       conda install -c anaconda ipykernel
       python -m ipykernel install --user --name=image-attributions

Now you can run the `example_attributions.ipynb`.  This assumes that you've stored attributions in `output_attributions.gzip` and the image segmentation prediction masks in `pred_masks.gzip`.

# 4 Citations

[Chen et al. 2019]
"TensorMask: A Foundation for Dense Object Segmentation".
Xinlei Chen, Ross Girshick, Kaiming He and Piotr Dollar.
(ICCV 2019)

[Liu et al. 2021]
"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
(CoRR, 2021)

[Kolesnicov et al. 2020]
"Big Transfer (BiT): General Visual Representation Learning"
Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.
(ECCV 2020)

[Sikdar et al. 2021]
"Integrated Directional Gradients: Feature Interaction Attribution for Neural NLP Models".
Sandipan Sikdar, Parantapa Bhattacharya, Kieran Heese.
(ACL-IJCNLP 2021)

[Sundararajan et al. 2017]
"Axiomatic Attribution for Deep Networks".
Mukund Sundararajan, Ankur Taly, and Qiqi Yan.
(CoRR, 2017)

[Wang et al. 2021]
"Sample-Efficient Neural Architecture Search by Learning Action Space for Monte Carlo Tree Search".
Linnan Wang, Saining Xie, Teng Lim, Rodrigo Fonseca, and Yuandong Tian.
(IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021)