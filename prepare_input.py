"""
Script for converting an input image into JSON.
"""

import mmcv
import json
import torch
import sys

def convert_img_to_json(img_path, save_path):
    """
    Converts an input image into JSON RGB representation.

    Parameters
    ----------
    img_path: str
        Path to the input image to prepare for modeling.
    save_path: str
        File location where final JSON output will be saved.

    Returns
    -------
    None
    """
    im = mmcv.imread(img_path)
    img_data = torch.tensor(im)
    
    json_dict = {
        "img_data": img_data.numpy().tolist(),
    }
    
    with open(save_path, "w") as fobj:
        json.dump(json_dict, fobj)
    return


def main():
    convert_img_to_json(str(sys.argv[1]), str(sys.argv[2]))

if __name__ == "__main__":
    main()
    
