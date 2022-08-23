import numpy as np


def m_iou(pred, mask):
    smooth = 1
    pred = pred.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()

    intersection = pred * mask
    union = pred + mask

    iou = (np.sum(intersection) + smooth) / (np.sum(union) - np.sum(intersection) + smooth)
    return float(iou)


def dice(pred, mask):
    smooth = 1
    pred = pred.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()

    pred_flat = np.reshape(pred, (-1))
    mask_flat = np.reshape(mask, (-1))
    intersection = pred_flat * mask_flat
    dice = (2. * intersection.sum() + smooth) / (pred.sum() + mask.sum() + smooth)
    return float(dice)
