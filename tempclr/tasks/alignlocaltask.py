import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import random

from ..utils.metrics import DTW_cum_dist, OTAM_cum_dist, cos_sim
from .task import Task


class AlignLocalTask(Task):
    def __call__(self, model, sample):
        alignment_loss_coeff = self.config.alignment.alignment_loss_coeff

        loss = None
        loss_scalar = float("inf")

        sample = self.reshape_subsample(sample)
        outputs = self.model(**sample)

        targets = sample["targets"].squeeze(0)
        non_empty_mask = targets.abs().sum(dim=1).bool()
        action_sequence = sample["sequence_targets"].squeeze(0)
        action_sequence = torch.unique_consecutive(action_sequence)

        pooled_text = outputs["pooled_text"]
        video_seq = outputs["video_seq"][non_empty_mask, :]

        non_empty_targets = torch.argmax(targets[non_empty_mask, :], dim=1)
        assert len(video_seq) == len(non_empty_targets)

        if model.training and len(action_sequence) > 1 and len(video_seq) > 1:
            alignment_distances = self.query_to_logits(
                pooled_text, video_seq, action_sequence, non_empty_targets
            )
            criterion = torch.nn.CrossEntropyLoss()
            alignment_loss = criterion(alignment_distances, torch.tensor([0]).cuda())
        else:
            alignment_loss = 0

        sample.update(outputs)
        if self.loss_fn is not None:
            loss = self.loss_fn(**sample)
            loss = (
                1 - alignment_loss_coeff
            ) * loss + alignment_loss_coeff * alignment_loss
            loss_scalar = loss.item()

        batch_size = sample["caps"].size(0)
        sample_size = 1
        return {
            "loss": loss,
            "loss_scalar": loss_scalar,
            "max_len": self.config.dataset.max_len,
            "batch_size": batch_size,
            "sample_size": sample_size,
        }

    def query_to_logits(self, query, keys, action_sequence, video_indices):
        # query: (32, 1, 4, 512), keys: (32, n, 4, 512)

        video_sampling_multiplier = self.config.alignment.video_sampling_multiplier
        inner_shuffling = self.config.alignment.inner_shuffling
        metric = self.config.alignment.metric
        total_sample_size = self.config.alignment.total_sample_size

        idx_f = np.linspace(
            0,
            keys.shape[-2] - 1,
            num=min(
                keys.shape[-2], action_sequence.shape[0] * video_sampling_multiplier
            ),
        )
        idxs = [int(f) for f in idx_f]
        indices = torch.tensor(idxs).cuda()
        keys = torch.index_select(keys, -2, indices).cuda()
        video_indices = torch.index_select(video_indices, 0, indices).cuda()

        # prepare for inner shuffling
        _, counts = torch.unique_consecutive(video_indices, return_counts=True)
        video_indices = torch.cumsum(
            torch.cat([torch.tensor([0]).cuda(), counts], dim=0), dim=0
        )
        assert keys.shape[-2] == video_indices[-1]

        sim = cos_sim(query, keys)
        origin_dists = 1 - sim

        dists = torch.index_select(origin_dists, 0, action_sequence)
        dists = rearrange(dists, "(b s) d -> b s d", b=1)

        negative_keys = []

        while len(negative_keys) < total_sample_size - 1:
            indexes = torch.randperm(action_sequence.shape[0])
            negative_action_sequence = action_sequence[indexes]

            if not torch.equal(action_sequence, negative_action_sequence):
                negative_key = torch.index_select(
                    origin_dists, 0, negative_action_sequence
                )

                if inner_shuffling:
                    origin_indices = [i for i in range(keys.shape[-2])]
                    inner_shuffled_indices = []
                    for i in range(len(video_indices) - 1):
                        step_indices = origin_indices[
                            video_indices[i] : video_indices[i + 1]
                        ]
                        random.shuffle(step_indices)
                        inner_shuffled_indices.extend(step_indices)
                    negative_key = torch.index_select(
                        negative_key, 1, torch.tensor(inner_shuffled_indices).cuda()
                    )

                negative_key = rearrange(negative_key, "(b s) d -> b s d", b=1)
                negative_keys.append(negative_key)

        dists = torch.stack([dists] + negative_keys, dim=1)

        if metric == "dtw":
            cum_dists = DTW_cum_dist(dists)
        elif metric == "otam":
            cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(
                rearrange(dists, "b n ts ss -> b n ss ts")
            )
        else:
            raise ValueError("unsupported metric (either dtw or otam)")

        logits = -cum_dists

        return logits
