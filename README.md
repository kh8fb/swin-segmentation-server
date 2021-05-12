# tensormask-segmentation-server
A cli-based server for obtaining image segmentation prediction masks from input images using `curl` requests.  This server utilizes the [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) state-of-the-art segmentation model.

### Installation

#### Initial Setup

This package requires the installation of both this repository as well as [Swin for Object Detection's Github](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) in an Anaconda environment.

First, create an Anaconda environment:

       conda create -n swin-segmentation-server python=3.8

Next, activate the environment, and `conda install` torch, torchvision, and cudatoolkit,

      conda activate swin-segmentation-server
      conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

#### Installing mmcv-full

Make sure that the path to cuda is available and set the `$CUDA_HOME` environmental variable. On some HPC servers this involves `module load gcccuda` and running `which nvcc` to obtain and set the `$CUDA_HOME` variable. **This server requires CUDA to run**

      export CUDA_HOME=/path/to/cuda-11.x.x

This next step depends on which cuda and PyTorch version you have installed.  Often, you can install a prebuilt package from mmcv-full with the following format:

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

Replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11` and `PyTorch 1.8.1`, use the following command:

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.1/index.html

See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.

#### Installing Swin

To install the Swin Models, `pip install` opencv and clone the [Swin for Object Detection Github Repo](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

   pip install opencv-python
   git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git 
   cd Swin-Transformer-Object-Detection
   pip install -r requirements/build.txt
   pip install -v -e .

#### Installing the swin-segmentation-server/

Finally, `cd` into this project's directory and install the requirements with

      cd swin-segmentation-server/
      pip install -e .
 
Now your environment is set up and you're ready to go.


### Usage

Activate the server directly from the command line with

	 swin-server -sp /path/to/swin_model.pth -cp /path/to/config_file.yaml

This command starts the server and load the model so that it's ready to go when called upon.

The finetuned Swin Transformer object detection model can be downloaded from this [Google drive folder](https://drive.google.com/drive/folders/1s4xvls62Z8uPAXW2jUu96Q2w1OinyEy6?usp=sharing)

You can provide additional arguments such as the hostname, port, and a cuda flag.

After the software has been started, run `curl` with the "model" filepath to get and download the attributions.

      curl http://localhost:8888/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"

#### Preparing Inputs for the Server

The `input_json_file.json` can be produced from an image with the script `prepare_input.py`. This will store the image as a JSON file of RGB values and the image can thus be passed to the server.

    python prepare_input.py /path/to/image.jpg input_json_file.json

### Interpreting Server Outputs

The prediction masks are stored in a dictionary with the key "pred_masks".  They are then compressed and able to be retrieved from the saved gzip file with:

      >>> import gzip
      >>> import torch
      >>> from io import BytesIO
      >>> with gzip.open("saved_file.gzip", 'rb') as fobj:
      >>>      x = BytesIO(fobj.read())
      >>>      preds_dict = torch.load(x)


### Running on a remote server

If you want to run swin-server on a remote server, you can specify the hostname to be 0.0.0.0 from the command line.  Then use the `hostname` command to find out which IP address the server is running on.

       swin-server -sp /path/to/swin_model.pth -cp /path/to/config.yaml -h 0.0.0.0 -p 8008
       hostname -I
       10.123.45.110 10.222.222.345 10.333.345.678

The first hostname result tells you which address to use in your `curl` request.

      curl http://10.123.45.110:8008/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"


### Model Results

This trained Swin Transformer model received the following results

| Dataset |   AP   |  AP50  |   APs  |   APm  |   APl  |
|---------|:------:|:------:|:------:|:------:|:------:|
|  Score  |  43.7  |  66.6  |  27.3  |  47.5  |   59   |


### Citations


@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}