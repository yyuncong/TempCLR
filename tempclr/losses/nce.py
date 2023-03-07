# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
softmax-based NCE loss, used by this project.
"""

import torch

from torch import nn

from .loss import Loss


class NCE(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, align_scores, **kargs):
        # note: we reuse the same shape as cls head in BERT (batch_size, 2)
        # but NCE only needs one logits.
        # (so we drop all weights in the second neg logits.)
        align_scores = align_scores[:, :1]
        # duplicate negative examples
        batch_size = align_scores.size(0) // 2
        pos_scores = align_scores[:batch_size]
        neg_scores = align_scores[batch_size:].view(1, batch_size).repeat(batch_size, 1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        return self.loss(
            scores,
            torch.zeros((batch_size,), dtype=torch.long, device=align_scores.device),
        )


class MMContraLoss(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kwargs):
        logits_per_video = pooled_video @ pooled_text.t()
        logits_per_text = pooled_text @ pooled_video.t()

        targets = torch.arange(
            pooled_video.size(0), dtype=torch.long, device=pooled_video.device
        )
        loss_video = self.loss(logits_per_video, targets)
        loss_text = self.loss(logits_per_text, targets)
        return loss_video + loss_text
