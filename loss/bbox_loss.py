import torch
import torch.nn.functional as F
import segmentation_models_pytorch
from network.models.SMP_DeepLab.utils import transpose_and_gather_output_array


def calculate_bbox_loss_without_heatmap(
    predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device
):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    object_boolean_mask = torch.zeros((flattened_index.shape), device="cuda")
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0 : int(num_objects[i])] = 1
    object_boolean_mask = (
        object_boolean_mask.unsqueeze(2).expand_as(predicted_bbox).float()
    )
    predicted_bbox, groundtruth_bbox = (
        predicted_bbox * object_boolean_mask,
        groundtruth_bbox * object_boolean_mask,
    )
    bbox_loss = F.smooth_l1_loss(predicted_bbox, groundtruth_bbox, reduction="sum")
    bbox_loss = bbox_loss / object_boolean_mask.sum()
    return bbox_loss


def calculate_bbox_loss_with_heatmap(
    predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device
):
    ################# DEBUG

    predicted_width = predicted_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=-1)
    predicted_height = predicted_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=-1)

    groundtruth_width = groundtruth_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=-1)
    groundtruth_height = groundtruth_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=-1)

    bbox_loss_width = torch.nn.functional.mse_loss(
        input=predicted_width.float(),
        target=groundtruth_width.float(),
        reduction="mean",
    )
    bbox_loss_height = torch.nn.functional.mse_loss(
        input=predicted_height.float(),
        target=groundtruth_height.float(),
        reduction="mean",
    )
    bbox_loss = bbox_loss_height + bbox_loss_width
    return bbox_loss
