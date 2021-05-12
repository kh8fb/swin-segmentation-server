"""
Run the model and get the pred_mask outputs.
"""

from mmdet.apis import inference_detector
import torch


def run_swin(img_array, model):
    """
    Run the model and receive the outputs

    Parameters
    ----------
    img_array: np.array(x,y,3)
        BGR array of the photo to obtain classification from.
    model: CascadeRCNN
        Model with the loaded pretrained states.
    Returns
    -------
    output: tuple
        Tuple with prediction boxes/confidence at index 0 and prediction masks at index 1.
    """
    result = inference_detector(model, img_array)
    return result


def get_pred_masks(result, threshold):
    """
    Extract image segment prediction masks where the prediction confidence exceeds threshold.

    Parameters
    ----------
    result: tuple
        Tuple with prediction boxes/confidence at index 0 and prediction masks at index 1.
    threshold: float
        Minimum threshold to include prediction masks in the final output. Value between 0 and 1.
    Returns
    -------
    pred_masks: torch.tensor(num_segments, height, width)
        Tensor with the prediction masks for each segment that passes the threshold.
    """
    classification_indices = [None]*len(result[0])
    for i, classification in enumerate(result[0]):
        num_classifications = 0
        if len(classification) > 0:
            for model_result in classification:
                if model_result[4] > 0.5: # passes a threshold of 0.5
                    num_classifications += 1
        classification_indices[i] = num_classifications

    tensor_list = []
    for (i, index) in enumerate(classification_indices):
        for j in range(index):
            tensor_list.append(torch.tensor(result[1][i][

    return torch.stack(tensor_list, dim=0)


def run_models(model_name, model, img, threshold):
    """
    Run the model on the input image and return the tensor of prediction masks for image segments.

    Parameters
    ----------
    model_name: str
        Name of the model that is being run.
        Currently supported is "swin".
    model: torch.nn.Module
        Model to run.
    img: np.array(x,y,3)
        BGR array of the photo to obtain segmentation from.
    threshold: float
        Minimum threshold to include prediction masks in the final output. Value between 0 and 1.
    Returns
    -------
    preds_dict: dict
        Dictionary containing the prediction masking tensors with the following keys:
            "pred_masks": torch.tensor(num_segmentations,height,width)
    """
    if model_name == "swin":
        result = run_swin(img, device)

        pred_masks = get_pred_masks(result, threshold)
        preds_dict = {"pred_masks": pred_masks}

    return preds_dict
