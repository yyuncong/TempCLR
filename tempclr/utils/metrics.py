import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np


def DTW_cum_dist(dists, lbda=0.1):
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    cum_dists[:, :, 0, 0] = dists[:, :, 0, 0]

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 0] = dists[:, :, l, 0] + cum_dists[:, :, l - 1, 0]

    for l in range(1, dists.shape[2]):
        for m in range(1, dists.shape[3]):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.logsumexp(
                torch.cat(
                    [
                        -cum_dists[:, :, l - 1, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l - 1, m].unsqueeze(dim=-1) / lbda,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

    return cum_dists[:, :, -1, -1]


def OTAM_cum_dist(dists, lbda=0.1):
    dists = F.pad(dists, (1, 1), "constant", 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] - lbda * torch.logsumexp(
            -cum_dists[:, :, 0, m - 1].unsqueeze(dim=-1) / lbda, dim=-1
        )

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.logsumexp(
            torch.cat(
                [
                    -cum_dists[:, :, l - 1, 0].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l - 1, 1].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l, 0].unsqueeze(dim=-1) / lbda,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.logsumexp(
                torch.cat(
                    [
                        -cum_dists[:, :, l - 1, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l, m - 1].unsqueeze(dim=-1) / lbda,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.logsumexp(
            torch.cat(
                [
                    -cum_dists[:, :, l - 1, -2].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l - 1, -1].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l, -2].unsqueeze(dim=-1) / lbda,
                ],
                dim=-1,
            ),
            dim=-1,
        )

    return cum_dists[:, :, -1, -1]


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def DTW_pad(dists, target_length0, target_length1):
    length_diff0 = target_length0 - dists.shape[-2]
    length_diff1 = target_length1 - dists.shape[-1]
    dists = F.pad(dists, (0, length_diff1, 0, length_diff0), "constant", float("inf"))
    dists[:, :, -length_diff0 - 1 :, -length_diff1:] = 0
    return dists


def OTAM_pad(dists, target_length0, target_length1):
    length_diff0 = target_length0 - dists.shape[-2]
    length_diff1 = target_length1 - dists.shape[-1]
    dists = F.pad(dists, (1, 1), "constant", 0)
    dists = F.pad(dists, (0, length_diff1, 0, length_diff0), "constant", float("inf"))
    dists[:, :, -length_diff0 - 1 :, -length_diff1:] = 0
    return dists


def DTW_eval_cum_dist(dists):
    cum_dists = torch.zeros(dists.shape, device=dists.device)
    cum_dists[:, :, 0, 0] = dists[:, :, 0, 0]

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]
    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 0] = dists[:, :, l, 0] + cum_dists[:, :, l - 1, 0]

    for l in range(1, dists.shape[2]):
        for m in range(1, dists.shape[3]):
            cum_dists[:, :, l, m] = dists[:, :, l, m] + torch.minimum(
                torch.minimum(cum_dists[:, :, l - 1, m - 1], cum_dists[:, :, l, m - 1]),
                cum_dists[:, :, l - 1, m],
            )

    return cum_dists


def OTAM_eval_cum_dist(dists):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    """
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - torch.minimum(
            torch.minimum(
                cum_dists[:, :, l - 1, 0],
                cum_dists[:, :, l - 1, 1],
            ),
            cum_dists[:, :, l, 0],
        )

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] + torch.minimum(
                cum_dists[:, :, l - 1, m - 1],
                cum_dists[:, :, l, m - 1],
            )

        cum_dists[:, :, l, -1] = dists[:, :, l, -1] + torch.minimum(
            torch.minimum(
                cum_dists[:, :, l - 1, -2],
                cum_dists[:, :, l - 1, -1],
            ),
            cum_dists[:, :, l, -2],
        )

    return cum_dists
