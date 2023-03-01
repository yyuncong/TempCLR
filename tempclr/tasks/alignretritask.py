import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import itertools
import random

from .. import models
from ..utils.metrics import DTW_cum_dist, OTAM_cum_dist, cos_sim
from .retritask import VideoRetriTask

class AlignRetriTask(VideoRetriTask):

    def build_model(self, checkpoint=None):
        if self.model is None:
            model_cls = getattr(models, self.config.model.model_cls)
            self.model = model_cls(self.config)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        if self.config.alignment.layernorm_only:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layer_name = name.split('.')[-2]
                    if layer_name != 'LayerNorm':
                        param.requires_grad = False
                        
        return self.model
    
    def __call__(self, model, sample):
        loss = None
        loss_scalar = float("inf")

        sample = self.reshape_subsample(sample)
        outputs = self.model(**sample)
        sample.update(outputs)

        pooled_video = sample['pooled_video']
        pooled_text = sample['pooled_text']
        text_start = sample['text_start']
        text_end = sample['text_end']
        video_start = sample['video_start']
        video_end = sample['video_end']

        alignment_loss = self.temporal_alignment(
            pooled_video, 
            pooled_text, 
            text_start, 
            text_end, 
            video_start,
            video_end,
        )
        alignment_loss_coeff = self.config.alignment.alignment_loss_coeff

        loss = self.loss_fn(**sample)
        loss = (1 - alignment_loss_coeff)*loss + alignment_loss_coeff*alignment_loss
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

    def temporal_alignment(self, pooled_video, pooled_text, text_start, text_end, video_start, video_end):
        clip_per_video = self.config.dataset.clip_per_video
        threshold = self.config.alignment.sampling_threshold
        total_sample_size = self.config.alignment.total_sample_size
        
        indices_list = [
            self.nonoverlapped_indices(
                text_start[i*clip_per_video: (i + 1)*clip_per_video], 
                text_end[i*clip_per_video: (i + 1)*clip_per_video],
                video_start[i*clip_per_video: (i + 1)*clip_per_video], 
                video_end[i*clip_per_video: (i + 1)*clip_per_video],
            ) 
            for i in range(pooled_video.shape[0]//clip_per_video)
        ]

        seq_lengths, filtered_video_count = self.sequence_length(indices_list, threshold)

        # if no qualified videos in the batch
        if filtered_video_count == 0:
            return torch.tensor(0)

        alignment_keys = [
            self.generate_alignment_keys(
                pooled_video[i*clip_per_video: (i + 1)*clip_per_video], 
                pooled_text[i*clip_per_video: (i + 1)*clip_per_video], 
                indices_list[i], 
                seq_lengths[i],
                threshold,
                total_sample_size,
            ) for i in range(pooled_video.shape[0]//clip_per_video)
        ]
        alignment_keys = [alignment_key for alignment_key in alignment_keys if alignment_key is not None]
        alignment_keys = torch.stack(alignment_keys, dim=1)
        alignment_keys = rearrange(alignment_keys, "b n l ts ss -> b (n l) ss ts")
        
        if self.config.alignment.metric == 'dtw':
            cum_dists = DTW_cum_dist(alignment_keys)
        elif self.config.alignment.metric == 'otam':
            cum_dists = OTAM_cum_dist(alignment_keys) + OTAM_cum_dist(
                rearrange(alignment_keys, "b n ts ss -> b n ss ts")
            )
        else:
            raise ValueError("unsupported metric (either dtw or otam)")
        logits = -cum_dists
        alignment_loss = self.generate_alignment_loss(logits, total_sample_size)

        return alignment_loss / filtered_video_count
    
    @staticmethod
    def nonoverlapped_indices(text_starts, text_ends, video_starts, video_ends):
        indices = []
        text_end = -1
        video_end = -1
        for i in range(len(text_starts)):
            if text_starts[i] > text_end and video_starts[i] > video_end:
                indices.append(i)
                text_end = text_ends[i]
                video_end = video_ends[i]
        return torch.tensor(indices).cuda()
    
    @staticmethod
    def sequence_length(indices_list, threshold):
        seq_lengths = []
        filtered_video_count = 0
        for indices in indices_list:
            length = len(indices)
            seq_lengths.append(length)
            if length >= threshold:
                filtered_video_count += 1
        return seq_lengths, filtered_video_count
    
    @staticmethod
    def generate_alignment_keys(pooled_video, pooled_text, indices, seq_length, threshold, total_sample_size):
        if seq_length < threshold:
            return None
        idx_f = np.linspace(0, len(indices) - 1, num=threshold)
        idxs = [int(f) for f in idx_f]
        sequence = indices[torch.tensor(idxs)]

        pooled_video = torch.index_select(pooled_video, 0, sequence)
        pooled_text = torch.index_select(pooled_text, 0, sequence)

        sim = cos_sim(pooled_text, pooled_video)
        orig_dists = 1 - sim
        dists = rearrange(orig_dists, "(b s) d -> b s d", b=1)

        negative_keys = []
        indices = [i for i in range(orig_dists.shape[-1])]
        perms = list(itertools.permutations(indices))
        random.shuffle(perms)
        indices = torch.tensor(indices).cuda()
        for perm in perms:
            negative_sequence = torch.tensor(perm).cuda()
            if not torch.equal(indices, negative_sequence):
                negative_key = torch.index_select(orig_dists, 0, negative_sequence)
                negative_key = rearrange(negative_key, "(b s) d -> b s d", b=1)
                negative_keys.append(negative_key)
            if len(negative_keys) >= total_sample_size - 1:
                break

        keys = torch.stack([dists] + negative_keys, dim=1)

        return keys
    
    @staticmethod
    def generate_alignment_loss(logits, total_sample_size):
        criterion = torch.nn.CrossEntropyLoss()
        alignment_losses = [
            criterion(logits[:, i*total_sample_size: (i + 1)*total_sample_size], torch.tensor([0]).cuda())
            for i in range(logits.shape[1]//total_sample_size)
        ]
        alignment_loss = sum(alignment_losses)
        return alignment_loss