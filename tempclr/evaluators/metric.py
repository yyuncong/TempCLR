# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import json
from itertools import groupby

from ..utils.metrics import DTW_eval_cum_dist, OTAM_eval_cum_dist, OTAM_pad, DTW_pad


class Metric(object):
    def __init__(self, config, metric_names):
        self.metric_names = metric_names

    def best_metric(self, metric):
        return metric[self.metric_names[0]]

    def save_metrics(self, fn, metrics):
        with open(fn, "w") as fw:
            json.dump(fw, metrics)

    def print_computed_metrics(self, metrics):
        raise NotImplementedError


class RetrievalMetric(Metric):
    """
    this is modified from `howto100m/metrics.py`.
    History of changes:
    refactor as a class.
    add metric_key in __init__
    """

    def __init__(self, config, metric_names=["R1", "R5", "R10", "MR"]):
        super().__init__(config, metric_names)
        self.error = False  # TODO(huxu): add to config to print error.

    def compute_metrics(self, outputs, texts, **kwargs):
        x = outputs
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
        d = d[:, np.newaxis]
        ind = sx - d
        ind = np.where(ind == 0)
        ind = ind[1]
        metrics = {}
        metrics["R1"] = float(np.sum(ind == 0)) / len(ind)
        metrics["R5"] = float(np.sum(ind < 5)) / len(ind)
        metrics["R10"] = float(np.sum(ind < 10)) / len(ind)
        metrics["MR"] = np.median(ind) + 1

        max_idx = np.argmax(outputs, axis=1)
        if self.error:
            # print top-20 errors.
            error = []
            for ex_idx in range(20):
                error.append((texts[ex_idx], texts[max_idx[ex_idx]]))
            metrics["error"] = error
        return metrics

    def print_computed_metrics(self, metrics):
        r1 = metrics["R1"]
        r5 = metrics["R5"]
        r10 = metrics["R10"]
        mr = metrics["MR"]
        print(
            "R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}".format(
                r1, r5, r10, mr
            )
        )
        if "error" in metrics:
            print(metrics["error"])


class FullVideoRetrievalMetric(Metric):
    """
    this is modified from `howto100m/metrics.py`.
    History of changes:
    refactor as a class.
    add metric_key in __init__
    """

    def __init__(
        self,
        config,
        metric_names=[
            "R1",
            "R5",
            "R10",
            "DTW-R1",
            "DTW-R5",
            "DTW-R10",
            "OTAM-R1",
            "OTAM-R5",
            "OTAM-R10",
        ],
    ):
        super().__init__(config, metric_names)
        self.error = False  # TODO(huxu): add to config to print error.

    def compute_metrics(self, outputs, texts, **kwargs):
        metrics = {}
        clip_ids = kwargs["video_ids"]
        m = outputs

        full_videos = list(set([d for d in clip_ids]))

        n_clips = len(clip_ids)
        n_vids = len(full_videos)
        clip_to_video = []
        [
            clip_to_video.extend([i for i, x in enumerate(full_videos) if x in clip_id])
            for clip_id in clip_ids
        ]
        clip_to_video = np.array(clip_to_video)

        video_inds = [
            [i for i, x in enumerate(clip_ids) if video in x] for video in full_videos
        ]  # list of length n_vids, with the corresponding clip_ind

        # Retrieval single | full caption -> full video | for each caption get max prediction over a video, then average over all captions of a video.
        video_preds_m = np.stack(
            [np.max(m[:, v], axis=1) for v in video_inds], 1
        )  # (n_clips, n_vids)
        video_preds_m2 = np.stack(
            [np.mean(video_preds_m[v, :], axis=0) for v in video_inds], 0
        )  # (n_vids, n_vids)
        metrics["R1"], metrics["R5"], metrics["R10"] = self.recall(
            video_preds_m2, np.arange(n_vids)
        )

        max_length = -float("inf")
        for video_ind in video_inds:
            max_length = max(max_length, len(video_ind))

        # Retrieval single | full caption -> full video | DTW
        clip_distances = torch.tensor(1 - m)
        all_matrices = []
        matrix_sizes = []
        for i in range(len(video_inds)):
            distance_matrices = []
            for j in range(len(video_inds)):
                distance_matrix = (
                    clip_distances[
                        video_inds[i][0] : video_inds[i][-1] + 1,
                        video_inds[j][0] : video_inds[j][-1] + 1,
                    ]
                    .view(1, 1, len(video_inds[i]), len(video_inds[j]))
                    .detach()
                )
                distance_matrix *= (max_length * max_length) / (
                    len(video_inds[i]) * len(video_inds[j])
                )
                distance_matrix_padded = DTW_pad(
                    distance_matrix, max_length, max_length
                )

                matrix_sizes.append(
                    torch.tensor([len(video_inds[i]), len(video_inds[j])]).unsqueeze(0)
                )

                distance_matrices.append(distance_matrix_padded)

            all_matrices.append(torch.cat(distance_matrices, dim=1))

        matrix_sizes = torch.cat(matrix_sizes).view(len(video_inds), len(video_inds), 2)
        all_matrices = torch.cat(all_matrices, dim=0)

        full_distance_matrix = DTW_eval_cum_dist(all_matrices)
        full_distance = full_distance_matrix[:, :, -1, -1]

        video_preds_alignment = (
            -full_distance.detach().cpu().numpy()
        )  # (n_vids, n_vids)
        metrics["DTW-R1"], metrics["DTW-R5"], metrics["DTW-R10"] = self.recall(
            video_preds_alignment, np.arange(n_vids)
        )

        # Retrieval single | full caption -> full video | OTAM
        clip_distances = torch.tensor(1 - m)
        video_inds = [
            [i for i, x in enumerate(clip_ids) if video in x] for video in full_videos
        ]  # list of length n_vids, with the corresponding clip_ind
        all_matrices = []
        all_matrices_t = []
        matrix_sizes = []
        for i in range(len(video_inds)):
            distance_matrices = []
            distance_matrices_t = []
            for j in range(len(video_inds)):
                distance_matrix = (
                    clip_distances[
                        video_inds[i][0] : video_inds[i][-1] + 1,
                        video_inds[j][0] : video_inds[j][-1] + 1,
                    ]
                    .view(1, 1, len(video_inds[i]), len(video_inds[j]))
                    .detach()
                )
                distance_matrix *= (max_length * max_length) / (
                    len(video_inds[i]) * len(video_inds[j])
                )

                matrix_sizes.append(
                    torch.tensor([len(video_inds[i]), len(video_inds[j])]).unsqueeze(0)
                )

                distance_matrix_padded = OTAM_pad(
                    distance_matrix, max_length, max_length
                )
                distance_matrix_padded_t = OTAM_pad(
                    torch.transpose(distance_matrix, dim0=-2, dim1=-1),
                    max_length,
                    max_length,
                )

                distance_matrices.append(distance_matrix_padded)
                distance_matrices_t.append(distance_matrix_padded_t)

            all_matrices.append(torch.cat(distance_matrices, dim=1))
            all_matrices_t.append(torch.cat(distance_matrices_t, dim=1))

        matrix_sizes = torch.cat(matrix_sizes).view(len(video_inds), len(video_inds), 2)
        all_matrices = torch.cat(all_matrices, dim=0)
        all_matrices_t = torch.cat(all_matrices_t, dim=0)
        full_distance_matrix = OTAM_eval_cum_dist(all_matrices)
        full_distance_matrix_t = OTAM_eval_cum_dist(all_matrices_t)
        full_distance = (
            full_distance_matrix[:, :, -1, -1] + full_distance_matrix_t[:, :, -1, -1]
        )

        video_preds_alignment = (
            -full_distance.detach().cpu().numpy()
        )  # (n_vids, n_vids)
        metrics["OTAM-R1"], metrics["OTAM-R5"], metrics["OTAM-R10"] = self.recall(
            video_preds_alignment, np.arange(n_vids)
        )

        return metrics

    def print_computed_metrics(self, metrics):
        r1 = metrics["R1"]
        r5 = metrics["R5"]
        r10 = metrics["R10"]

        dtw_r1 = metrics["DTW-R1"]
        dtw_r5 = metrics["DTW-R5"]
        dtw_r10 = metrics["DTW-R10"]

        otam_r1 = metrics["OTAM-R1"]
        otam_r5 = metrics["OTAM-R5"]
        otam_r10 = metrics["OTAM-R10"]

        print(
            "Caption Average:\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}\nDTW:\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}\nOTAM\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}".format(
                r1,
                r5,
                r10,
                dtw_r1,
                dtw_r5,
                dtw_r10,
                otam_r1,
                otam_r5,
                otam_r10,
            )
        )
        if "error" in metrics:
            print(metrics["error"])

    @staticmethod
    def recall(mat, gts):
        # mat is of shape (Queries, Targets), where higher=prediction
        # gts is of shape (Queries, )

        predictions = np.argsort(mat, 1)  # (Queries, Targets)
        top_1 = predictions[:, -1]

        recall = np.mean(top_1 == gts)
        recall_top5 = np.mean([l in p for l, p in zip(gts, predictions[:, -5:])])
        recall_top10 = np.mean([l in p for l, p in zip(gts, predictions[:, -10:])])

        return recall, recall_top5, recall_top10


class FullVideoBGRetrievalMetric(Metric):
    """
    this is modified from `howto100m/metrics.py`.
    History of changes:
    refactor as a class.
    add metric_key in __init__
    """

    def __init__(
        self,
        config,
        metric_names=[
            "R1",
            "R5",
            "R10",
            "DTW-R1",
            "DTW-R5",
            "DTW-R10",
            "Combined-DTW-R1",
            "Combined-DTW-R5",
            "Combined-DTW-R10",
            "OTAM-R1",
            "OTAM-R5",
            "OTAM-R10",
            "Combined-OTAM-R1",
            "Combined-OTAM-R5",
            "Combined-OTAM-R10",
        ],
    ):
        super().__init__(config, metric_names)
        self.error = False  # TODO(huxu): add to config to print error.

    def compute_metrics(self, outputs, texts, **kwargs):
        metrics = {}
        combine_coeff = 0.012
        clip_ids = kwargs["video_ids"]
        text_lengths = kwargs["text_lengths"]
        video_lengths = kwargs["video_lengths"]
        m = outputs

        video_unique_ids = [i[0] for i in groupby(clip_ids)]
        text_clip_ids = []
        [
            text_clip_ids.extend([video_unique_ids[i]] * text_lengths[i])
            for i in range(len(video_unique_ids))
        ]
        video_clip_ids = []
        [
            video_clip_ids.extend([video_unique_ids[i]] * video_lengths[i])
            for i in range(len(video_unique_ids))
        ]

        full_videos = sorted(list(set([d for d in clip_ids])))

        n_clips = len(clip_ids)
        n_vids = len(full_videos)
        text_clip_to_video = []
        [
            text_clip_to_video.extend(
                [i for i, x in enumerate(full_videos) if x in clip_id]
            )
            for clip_id in text_clip_ids
        ]
        text_clip_to_video = np.array(text_clip_to_video)

        video_clip_to_video = []
        [
            video_clip_to_video.extend(
                [i for i, x in enumerate(full_videos) if x in clip_id]
            )
            for clip_id in video_clip_ids
        ]
        video_clip_to_video = np.array(video_clip_to_video)

        text_inds = [
            [i for i, x in enumerate(text_clip_ids) if video in x]
            for video in full_videos
        ]  # list of length n_vids, with the corresponding clip_ind
        video_inds = [
            [i for i, x in enumerate(video_clip_ids) if video in x]
            for video in full_videos
        ]

        # Retrieval single | full caption -> full video | for each caption get max prediction over a video, then average over all captions of a video.
        video_preds_m = np.stack(
            [np.max(m[:, v], axis=1) for v in video_inds], 1
        )  # (n_clips, n_vids)
        video_preds_m2 = np.stack(
            [np.mean(video_preds_m[v, :], axis=0) for v in text_inds], 0
        )  # (n_vids, n_vids)
        metrics["R1"], metrics["R5"], metrics["R10"] = self.recall(
            video_preds_m2, np.arange(n_vids)
        )

        max_video_length = -float("inf")
        for video_ind in video_inds:
            max_video_length = max(max_video_length, len(video_ind))

        max_text_length = -float("inf")
        for text_ind in text_inds:
            max_text_length = max(max_text_length, len(text_ind))

        # Retrieval single | full caption -> full video | DTW
        clip_distances = torch.tensor(1 - m)
        all_matrices = []
        matrix_sizes = []
        for i in range(len(text_inds)):
            distance_matrices = []
            for j in range(len(video_inds)):
                distance_matrix = (
                    clip_distances[
                        text_inds[i][0] : text_inds[i][-1] + 1,
                        video_inds[j][0] : video_inds[j][-1] + 1,
                    ]
                    .view(1, 1, len(text_inds[i]), len(video_inds[j]))
                    .detach()
                )

                min_across_captions, _ = torch.min(
                    distance_matrix.view(len(text_inds[i]), len(video_inds[j])), dim=0
                )
                indices = torch.argsort(min_across_captions)[
                    : int(1.3 * distance_matrix.shape[-2])
                ]
                indices, _ = torch.sort(indices)
                indices = torch.unique(indices)
                distance_matrix = torch.index_select(
                    distance_matrix, dim=-1, index=indices
                )

                distance_matrix *= (max_text_length * max_video_length) / (
                    distance_matrix.shape[-2] * distance_matrix.shape[-1]
                )
                distance_matrix_padded = DTW_pad(
                    distance_matrix, max_text_length, max_video_length
                )

                matrix_sizes.append(
                    torch.tensor([len(text_inds[i]), len(video_inds[j])]).unsqueeze(0)
                )

                distance_matrices.append(distance_matrix_padded)

            all_matrices.append(torch.cat(distance_matrices, dim=1))

        matrix_sizes = torch.cat(matrix_sizes).view(len(text_inds), len(video_inds), 2)
        all_matrices = torch.cat(all_matrices, dim=0)

        full_distance_matrix = DTW_eval_cum_dist(all_matrices)
        full_distance = full_distance_matrix[:, :, -1, -1]

        video_preds_alignment = (
            -full_distance.detach().cpu().numpy()
        )  # (n_vids, n_vids)
        metrics["DTW-R1"], metrics["DTW-R5"], metrics["DTW-R10"] = self.recall(
            video_preds_alignment, np.arange(n_vids)
        )

        # Retrieval single | full caption -> full video | OTAM
        clip_distances = torch.tensor(1 - m)
        all_matrices = []
        all_matrices_t = []
        matrix_sizes = []
        for i in range(len(video_inds)):
            distance_matrices = []
            distance_matrices_t = []
            for j in range(len(video_inds)):
                distance_matrix = (
                    clip_distances[
                        text_inds[i][0] : text_inds[i][-1] + 1,
                        video_inds[j][0] : video_inds[j][-1] + 1,
                    ]
                    .view(1, 1, len(text_inds[i]), len(video_inds[j]))
                    .detach()
                )

                min_across_captions, _ = torch.min(
                    distance_matrix.view(len(text_inds[i]), len(video_inds[j])), dim=0
                )
                indices = torch.argsort(min_across_captions)[
                    : int(1.3 * distance_matrix.shape[-2])
                ]
                indices, _ = torch.sort(indices)
                indices = torch.unique(indices)
                distance_matrix = torch.index_select(
                    distance_matrix, dim=-1, index=indices
                )

                distance_matrix *= (max_text_length * max_video_length) / (
                    distance_matrix.shape[-2] * distance_matrix.shape[-1]
                )

                matrix_sizes.append(
                    torch.tensor([len(text_inds[i]), len(video_inds[j])]).unsqueeze(0)
                )

                distance_matrix_padded = OTAM_pad(
                    distance_matrix, max_text_length, max_video_length
                )
                distance_matrix_padded_t = OTAM_pad(
                    torch.transpose(distance_matrix, dim0=-2, dim1=-1),
                    max_text_length,
                    max_video_length,
                )

                distance_matrices.append(distance_matrix_padded)
                distance_matrices_t.append(distance_matrix_padded_t)

            all_matrices.append(torch.cat(distance_matrices, dim=1))
            all_matrices_t.append(torch.cat(distance_matrices_t, dim=1))

        matrix_sizes = torch.cat(matrix_sizes).view(len(text_inds), len(video_inds), 2)
        all_matrices = torch.cat(all_matrices, dim=0)
        all_matrices_t = torch.cat(all_matrices_t, dim=0)
        full_distance_matrix = OTAM_eval_cum_dist(all_matrices)
        full_distance_matrix_t = OTAM_eval_cum_dist(all_matrices_t)
        full_distance = (
            full_distance_matrix[:, :, -1, -1] + full_distance_matrix_t[:, :, -1, -1]
        )

        video_preds_alignment = (
            -full_distance.detach().cpu().numpy()
        )  # (n_vids, n_vids)
        metrics["OTAM-R1"], metrics["OTAM-R5"], metrics["OTAM-R10"] = self.recall(
            video_preds_alignment, np.arange(n_vids)
        )
        return metrics

    def print_computed_metrics(self, metrics):
        r1 = metrics["R1"]
        r5 = metrics["R5"]
        r10 = metrics["R10"]

        dtw_r1 = metrics["DTW-R1"]
        dtw_r5 = metrics["DTW-R5"]
        dtw_r10 = metrics["DTW-R10"]

        otam_r1 = metrics["OTAM-R1"]
        otam_r5 = metrics["OTAM-R5"]
        otam_r10 = metrics["OTAM-R10"]

        print(
            "Caption Average:\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}\nDTW:\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}\nOTAM\nR@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}".format(
                r1,
                r5,
                r10,
                dtw_r1,
                dtw_r5,
                dtw_r10,
                otam_r1,
                otam_r5,
                otam_r10,
            )
        )
        if "error" in metrics:
            print(metrics["error"])

    @staticmethod
    def recall(mat, gts):
        # mat is of shape (Queries, Targets), where higher=prediction
        # gts is of shape (Queries, )

        predictions = np.argsort(mat, 1)  # (Queries, Targets)
        top_1 = predictions[:, -1]

        recall = np.mean(top_1 == gts)
        recall_top5 = np.mean([l in p for l, p in zip(gts, predictions[:, -5:])])
        recall_top10 = np.mean([l in p for l, p in zip(gts, predictions[:, -10:])])

        return recall, recall_top5, recall_top10


class CrossTaskMetric(Metric):
    def __init__(self, config, metric_names=["recall"]):
        super().__init__(config, metric_names)

    def compute_metrics(self, outputs, targets, **kwargs):
        """refactored from line 166:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py"""

        recalls = self._get_recalls(Y_true=targets, Y_pred=outputs)
        results = {}
        for task, rec in recalls.items():
            results[str(task)] = rec

        avg_recall = np.mean(list(recalls.values()))
        results["recall"] = avg_recall
        return results

    def print_computed_metrics(self, metrics):
        print("Recall: {0:0.3f}".format(metrics["recall"]))
        for task in metrics:
            if task != "recall":
                print("Task {0}. Recall = {1:0.3f}".format(task, metrics[task]))

    def _get_recalls(self, Y_true, Y_pred):
        """refactored from
        https://github.com/DmZhukov/CrossTask/blob/master/train.py"""

        step_match = {task: 0 for task in Y_true.keys()}
        step_total = {task: 0 for task in Y_true.keys()}
        for task, ys_true in Y_true.items():
            ys_pred = Y_pred[task]
            for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
                y_true = ys_true[vid]
                y_pred = ys_pred[vid]
                step_total[task] += (y_true.sum(axis=0) > 0).sum()
                step_match[task] += (y_true * y_pred).sum()
        recalls = {task: step_match[task] / n for task, n in step_total.items()}
        return recalls
