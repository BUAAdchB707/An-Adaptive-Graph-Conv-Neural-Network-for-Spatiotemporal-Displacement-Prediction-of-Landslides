import torch
import numpy as np


def get_metrics(df_pred, df_test, mask=None):
    """
    Calculate the MAE, RMSE, MAPE, R2
    """
    assert df_pred.shape == df_test.shape
    mape = masked_mape(df_pred, df_test, mask).item()
    mae = masked_mae(df_pred, df_test, mask).item()
    rmse = masked_rmse(df_pred, df_test, mask).item()
    # r2 = masked_r2(df_pred, df_test, mask).item()
    header = " MAE: %7.4f  RMSE: %7.4f  MAPE: %7.4f  "
    return (mae, rmse, mape), header


def masked_mae(preds, labels, mask=None):
    if mask is None:
        mask = ~torch.isnan(labels)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.subtract(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mean(x, mask=None):
    if mask is None:
        mask = ~torch.isnan(x)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mean = x * mask
    mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
    return torch.mean(mean)


def masked_r2(preds, labels, mask=None):
    r = masked_mse(preds, labels, mask) / masked_mse(masked_mean(labels, mask), labels, mask)
    return 1. - r


def masked_mse(preds, labels, mask=None):
    if mask is None:
        mask = ~torch.isnan(labels)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.subtract(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, mask=None):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, mask=mask))


def masked_mape(preds, labels, mask):
    if mask is None:
        mask = ~torch.isnan(labels)
    mask *= ~torch.isclose(labels, torch.tensor(0.).expand_as(labels).to(labels.device), atol=5e-5, rtol=0.)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100
