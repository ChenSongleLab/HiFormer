
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops




def box_cxcywh_to_xyxy(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)



def box_xyxy_to_vextex(x):
    v1 = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=-1).unsqueeze(2)
    v2 = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 5]], dim=-1).unsqueeze(2)
    v3 = torch.stack([x[:, :, 0], x[:, :, 4], x[:, :, 2]], dim=-1).unsqueeze(2)
    v4 = torch.stack([x[:, :, 0], x[:, :, 4], x[:, :, 5]], dim=-1).unsqueeze(2)
    v5 = torch.stack([x[:, :, 3], x[:, :, 1], x[:, :, 2]], dim=-1).unsqueeze(2)
    v6 = torch.stack([x[:, :, 3], x[:, :, 1], x[:, :, 5]], dim=-1).unsqueeze(2)
    v7 = torch.stack([x[:, :, 3], x[:, :, 4], x[:, :, 2]], dim=-1).unsqueeze(2)
    v8 = torch.stack([x[:, :, 3], x[:, :, 4], x[:, :, 5]], dim=-1).unsqueeze(2)

    v = torch.cat([v1,v2,v3,v4,v5,v6,v7,v8], dim=2)

    return v





def box_12_lines(x):
    l = []
    index_v = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for v in range(len(index_v)):
        i = index_v[v][0]
        j = index_v[v][1]
        l.append([x[:, :, i, 0], x[:, :, j, 0]], [x[:, :, i, 1], x[:, :, j, 1]], [x[:, :, i, 2], x[:, :, j, 2]])
    return l


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """



    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

def box_transform(boxes):

    boxes[:, 3:] += boxes[:, :3]
    boxes = box_xyxy_to_cxcywh(boxes)
    return boxes





















def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]



    union = area1[:, None] + area2 - inter + 1e-8

    iou = inter / union
    return iou, union


def box_iou_match(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :3], boxes2[:, :3])
    rb = torch.min(boxes1[:,  3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1] * wh[:, 2]



    union = area1 + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """



    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]




    area = area + 1e-8
    return iou - (area - union) / area

def generalized_box_iou_match(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """



    assert (boxes1[:, 3:] > boxes1[:, :3]).all()
    assert (boxes2[:, 3:] > boxes2[:, :3]).all()
    iou, union = box_iou_match(boxes1, boxes2)

    lt = torch.min(boxes1[:, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:,  3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1] * wh[:, 2]





    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
