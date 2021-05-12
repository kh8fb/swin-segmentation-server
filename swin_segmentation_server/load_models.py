"""
Interface for loading the supported image segmentation models.
"""

from mmdet.apis import init_detector

def load_swin_model(model_path, cfg_path):
    """
    Load the pretrained Swin Transformer model states and prepare the model for image segmentation.

    Paramters
    ---------
    model_path: str
        Path to the pretrained model states binary file.
    cfg_path: str
        Path to the model's Configuration file.  Located in the .configs folder.

    Returns
    -------
    model: CascadeRCNN
        Model with the loaded pretrained states.
    """
    # set up model config
    model = init_detector(cfg_path, model_path, device='cuda:0')
    return model

def load_models(swin_path, cfg_path):
    """
    Load the model return them in a dictionary.

    Parameters
    ----------
    swin_path: str or None
        Path to the pretrained Swin Transformer model states binary file.
    cfg_path: str
        Path to the model's configuration file.  Located in the .configs folder.

    Returns
    -------
    model_dict: dict
        Dictionary storing the model and model name.
        Current keys are 'model_name', 'model'.
    """
    if swin_path is not None:
        swin_model = load_swin_model(str(swin_path), str(cfg_path))
        return {"model_name": "swin", "model": swin_model}
    # add additional models here
    else:
        return None
