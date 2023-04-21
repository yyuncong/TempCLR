# Copyright (c) Facebook, Inc. All Rights Reserved

"""
Processors for all downstream (ds) tasks.
"""

import json
import os
import pickle
import random
import math
import numpy as np
import torch

from collections import defaultdict

from .processor import (
    MetaProcessor,
    VideoProcessor,
    TextProcessor,
    Aligner,
    MMAttentionMask2DProcessor,
)

from .how2processor import TextGenerationProcessor


# ------------- A General Aligner for all downstream tasks-----------------


class DSAligner(Aligner):
    """
    Downstream (DS) aligner shared by all datasets.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        # random sample a starting sec for video.
        video_start = 0
        video_end = min(len(video_feature), self.max_video_len)
        # the whole sequence is a single clip.
        video_clips = {"start": [video_start], "end": [video_end]}

        text_feature = {
            "cap": [text_feature],
            "start": [video_start],
            "end": [len(text_feature) / wps],
        }
        text_clip_indexs = [0]

        vfeats, vmasks = self._build_video_seq(video_feature, video_clips)
        caps, cmasks = self._build_text_seq(text_feature, text_clip_indexs)

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_id": video_id,
        }


# -------------------- Youcook -----------------------


class YoucookMetaProcessor(MetaProcessor):
    """Youcook dataset.
    reference: `howto100m/youcook_dataloader.py`
    note that the data can be different as the
    (1) some videos already in Howto100m are removed.
    (2) stop words are removed from caption
    TODO (huxu): make a flag to load the original caption.
    (see youcookii_annotations_trainval.json).

    The max_video_len can be 264 and text can be 64 tokens.
    In reality we may not need that long. see projects/task/youcook.yaml
    """

    def __init__(self, config):
        super().__init__(config)
        vfeat_dir = config.vfeat_dir
        print(self._get_split_path(config))
        with open(self._get_split_path(config), "rb") as fd:
            data = pickle.load(fd)
            all_valid_video_ids = set(
                [os.path.splitext(fn)[0] for fn in os.listdir(vfeat_dir)]
            )
            recs = []
            video_ids = set()
            valid_video_ids = set()
            for rec in data:  # filter videos not available.
                udl_idx = rec["id"].rindex("_")
                video_id = rec["id"][:udl_idx]
                video_ids.add(video_id)
                if video_id in all_valid_video_ids:
                    valid_video_ids.add(video_id)
                    recs.append(rec)
            print("total video_ids in .pkl", len(video_ids))
            print("valid video_ids in .pkl", len(valid_video_ids))
            print("please verify {train,val}_list.txt")
            data = recs
            self.data = data

        with open(config.trainval_annotation) as fd:
            self.youcook_annotation = json.load(fd)["database"]
        if config.use_annotation_text is True:
            print("using text in annotation.")
            self.use_annotation_caption = True
        else:
            self.use_annotation_caption = False

    def __getitem__(self, idx):
        def _get_video_and_caption(rec):
            vid = rec["id"]
            udl_idx = vid.rindex("_")
            video_id, clip_id = vid[:udl_idx], int(vid[udl_idx + 1 :])
            clip = self.youcook_annotation[video_id]["annotations"][clip_id]
            start, end = clip["segment"]
            if self.use_annotation_caption:
                caption = clip["sentence"]
            else:
                caption = rec["caption"]
            return (video_id, start, end), caption

        rec = self.data[idx]
        video_info, text_info = _get_video_and_caption(rec)
        return video_info, text_info


class YoucookVideoProcessor(VideoProcessor):
    """video_fn is a tuple of (video_id, start, end) now."""

    def __call__(self, video_fn):
        video_id, start, end = video_fn
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        return feat[start:end]


class YoucookFullVideoBGProcessor(VideoProcessor):
    """video_fn is video_id now."""

    def __call__(self, video_fn):
        video_id = video_fn
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        return feat


class YoucookFullVideoBGTextProcessor(TextProcessor):
    def __call__(self, text_ids):
        tokens = []
        for text_id in text_ids:
            caption = self.tokenizer(text_id, add_special_tokens=False)
            tokens.append(caption["input_ids"])
        return tokens


class YoucookFullVideoBGAligner(Aligner):
    def __init__(self, config):
        super().__init__(config)
        self.sliding_window = config.sliding_window
        self.sliding_window_size = config.sliding_window_size

    def __call__(self, video_id, video_feature, text_feature):
        vid = video_id
        video_len = len(video_feature)

        vfeats, vmasks = [], []

        for window_start in range(0, video_len, self.sliding_window):
            video_start = 0
            video_end = min(video_len - window_start, self.sliding_window_size)
            video_clip = {"start": [video_start], "end": [video_end]}

            vfeat, vmask = self._build_video_seq(
                video_feature[window_start : window_start + video_end], video_clip
            )
            vfeats.append(vfeat)
            vmasks.append(vmask)

            if (video_len - window_start) <= self.sliding_window_size:
                break

        vfeats = torch.stack(vfeats)
        vmasks = torch.stack(vmasks)

        caps, cmasks = [], []
        for step in text_feature:
            step_text_feature = {"start": [0], "end": [1], "cap": [step]}
            step_text_clip_index = [0]
            cap, cmask = self._build_text_seq(step_text_feature, step_text_clip_index)
            caps.append(cap)
            cmasks.append(cmask)
        caps = torch.stack(caps)
        cmasks = torch.stack(cmasks)

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,  # X for original code.
            "vmasks": vmasks,
            "video_id": vid,
            "video_len": video_len,  # for later checking.
        }


class YoucookFullVideoBGMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        vfeat_dir = config.vfeat_dir
        print(self._get_split_path(config))
        with open(self._get_split_path(config), "rb") as fd:
            data = pickle.load(fd)
            all_valid_video_ids = set(
                [os.path.splitext(fn)[0] for fn in os.listdir(vfeat_dir)]
            )
            recs = defaultdict(list)
            video_ids = set()
            valid_video_ids = set()
            for rec in data:  # filter videos not available.
                udl_idx = rec["id"].rindex("_")
                video_id = rec["id"][:udl_idx]
                video_ids.add(video_id)
                if video_id in all_valid_video_ids:
                    valid_video_ids.add(video_id)
                    recs[video_id].append(rec)
            print("total video_ids in .pkl", len(video_ids))
            print("valid video_ids in .pkl", len(valid_video_ids))
            print("please verify {train,val}_list.txt")
            self.data = list(valid_video_ids)
            self.recs = recs

        with open(config.trainval_annotation) as fd:
            self.youcook_annotation = json.load(fd)["database"]
        if config.use_annotation_text is True:
            print("using text in annotation.")
            self.use_annotation_caption = True
        else:
            self.use_annotation_caption = False

    def __getitem__(self, idx):
        def _get_captions(recs):
            captions = []
            for rec in recs:
                vid = rec["id"]
                udl_idx = vid.rindex("_")
                video_id, clip_id = vid[:udl_idx], int(vid[udl_idx + 1 :])
                clip = self.youcook_annotation[video_id]["annotations"][clip_id]
                if self.use_annotation_caption:
                    caption = clip["sentence"]
                else:
                    caption = rec["caption"]
                captions.append(caption)
            return captions

        video_id = self.data[idx]
        return video_id, _get_captions(self.recs[video_id])


# --------------------- CrossTask -------------------------


class CrossTaskMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        np.random.seed(0)  # deterministic random split.
        task_vids = self._get_vids(
            config.train_csv_path, config.vfeat_dir, config.annotation_path
        )

        val_vids = self._get_vids(
            config.val_csv_path, config.vfeat_dir, config.annotation_path
        )

        # filter out those task and vids appear in val_vids.
        task_vids = {
            task: [
                vid for vid in vids if task not in val_vids or vid not in val_vids[task]
            ]
            for task, vids in task_vids.items()
        }

        primary_info = self._read_task_info(config.primary_path)
        test_tasks = set(primary_info["steps"].keys())

        # if args.use_related:
        related_info = self._read_task_info(config.related_path)
        task_steps = {**primary_info["steps"], **related_info["steps"]}
        n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
        # else:
        #     task_steps = primary_info['steps']
        #     n_steps = primary_info['n_steps']
        all_tasks = set(n_steps.keys())
        # filter and keep task in primary or related.
        task_vids = {
            task: vids for task, vids in task_vids.items() if task in all_tasks
        }
        # vocab-by-step matrix (A) and vocab (M)
        # we do not use BoW.
        # A, M = self._get_A(task_steps, share="words")

        train_vids, test_vids = self._random_split(
            task_vids, test_tasks, config.n_train
        )
        print("train_num_videos", sum(len(vids) for vids in train_vids.values()))
        print("test_num_videos", sum(len(vids) for vids in test_vids.values()))
        # added by huxu to automatically determine the split.
        split_map = {"train": train_vids, "valid": test_vids, "test": test_vids}
        task_vids = split_map[config.split]

        self.vids = []
        for task, vids in task_vids.items():
            self.vids.extend([(task, vid) for vid in vids])
        self.task_steps = task_steps
        self.n_steps = n_steps

    def __getitem__(self, idx):
        task, vid = self.vids[idx]
        n_steps = self.n_steps[task]
        steps = self.task_steps[task]
        assert len(steps) == n_steps
        return (task, vid, steps, n_steps), (task, vid, steps, n_steps)

    def __len__(self):
        return len(self.vids)

    def _random_split(self, task_vids, test_tasks, n_train):
        train_vids = {}
        test_vids = {}
        for task, vids in task_vids.items():
            if task in test_tasks and len(vids) > n_train:
                train_vids[task] = np.random.choice(
                    vids, n_train, replace=False
                ).tolist()
                test_vids[task] = [vid for vid in vids if vid not in train_vids[task]]
            else:
                train_vids[task] = vids
        return train_vids, test_vids

    def _get_vids(self, path, vfeat_dir, annotation_path):
        """refactored from
        https://github.com/DmZhukov/CrossTask/blob/master/data.py
        changes: add `vfeat_dir` to check if the video is available.
        add `annotation_path` to check if the video is available.
        """

        task_vids = {}
        with open(path, "r") as f:
            for line in f:
                task, vid, url = line.strip().split(",")
                # double check the video is available.
                if not os.path.exists(os.path.join(vfeat_dir, vid + ".npy")):
                    continue
                # double check the annotation is available.
                if not os.path.exists(
                    os.path.join(annotation_path, task + "_" + vid + ".csv")
                ):
                    continue
                if task not in task_vids:
                    task_vids[task] = []
                task_vids[task].append(vid)
        return task_vids

    def _read_task_info(self, path):
        titles = {}
        urls = {}
        n_steps = {}
        steps = {}
        with open(path, "r") as f:
            idx = f.readline()
            while idx != "":
                idx = idx.strip()
                titles[idx] = f.readline().strip()
                urls[idx] = f.readline().strip()
                n_steps[idx] = int(f.readline().strip())
                steps[idx] = f.readline().strip().split(",")
                next(f)
                idx = f.readline()
        return {"title": titles, "url": urls, "n_steps": n_steps, "steps": steps}

    def _get_A(self, task_steps, share="words"):
        raise ValueError("running get_A is not allowed for BERT.")
        """Step-to-component matrices."""
        if share == "words":
            # share words
            task_step_comps = {
                task: [step.split(" ") for step in steps]
                for task, steps in task_steps.items()
            }
        elif share == "task_words":
            # share words within same task
            task_step_comps = {
                task: [[task + "_" + tok for tok in step.split(" ")] for step in steps]
                for task, steps in task_steps.items()
            }
        elif share == "steps":
            # share whole step descriptions
            task_step_comps = {
                task: [[step] for step in steps] for task, steps in task_steps.items()
            }
        else:
            # no sharing
            task_step_comps = {
                task: [[task + "_" + step] for step in steps]
                for task, steps in task_steps.items()
            }
        # BERT tokenizer here?
        vocab = []
        for task, steps in task_step_comps.items():
            for step in steps:
                vocab.extend(step)
        vocab = {comp: m for m, comp in enumerate(set(vocab))}
        M = len(vocab)
        A = {}
        for task, steps in task_step_comps.items():
            K = len(steps)
            a = torch.zeros(M, K)
            for k, step in enumerate(steps):
                a[[vocab[comp] for comp in step], k] = 1
            a /= a.sum(dim=0)
            A[task] = a
        return A, M


class CrossTaskVideoProcessor(VideoProcessor):
    def __call__(self, video_fn):
        task, vid, steps, n_steps = video_fn
        video_fn = os.path.join(self.vfeat_dir, vid + ".npy")
        feat = np.load(video_fn)
        return feat


class CrossTaskTextProcessor(TextProcessor):
    def __call__(self, text_id):
        task, vid, steps, n_steps = text_id
        step_ids = []
        for step_str in steps:
            step_ids.append(
                self.tokenizer(step_str, add_special_tokens=False)["input_ids"]
            )
        return step_ids


class CrossTaskAligner(Aligner):
    """
    TODO: it's not clear yet the formulation of the task; finish this later.
    """

    def __init__(self, config):
        super().__init__(config)
        self.annotation_path = config.annotation_path
        self.sliding_window = config.sliding_window
        self.sliding_window_size = config.sliding_window_size

    def __call__(self, video_id, video_feature, text_feature):
        task, vid, steps, n_steps = video_id
        annot_path = os.path.join(self.annotation_path, task + "_" + vid + ".csv")
        video_len = len(video_feature)

        labels = torch.from_numpy(
            self._read_assignment(video_len, n_steps, annot_path)
        ).float()

        sequence_labels = self._read_sequence(annot_path).long()

        vfeats, vmasks, targets = [], [], []
        # sliding window on video features and targets.
        for window_start in range(0, video_len, self.sliding_window):
            video_start = 0
            video_end = min(video_len - window_start, self.sliding_window_size)
            video_clip = {"start": [video_start], "end": [video_end]}

            vfeat, vmask = self._build_video_seq(
                video_feature[window_start : window_start + video_end], video_clip
            )

            target = labels[window_start : window_start + video_end]
            assert len(vfeat) >= len(target), "{},{}".format(len(vfeat), len(target))
            # TODO: randomly drop all zero targets for training ?
            # if self.split == "train" and target.sum() == 0:
            #     continue
            vfeats.append(vfeat)
            vmasks.append(vmask)
            targets.append(target)

            if (video_len - window_start) <= self.sliding_window_size:
                break

        vfeats = torch.stack(vfeats)
        vmasks = torch.stack(vmasks)
        targets = torch.cat(targets, dim=0)

        caps, cmasks = [], []
        for step in text_feature:
            step_text_feature = {"start": [0], "end": [1], "cap": [step]}
            step_text_clip_index = [0]
            cap, cmask = self._build_text_seq(step_text_feature, step_text_clip_index)
            caps.append(cap)
            cmasks.append(cmask)
        caps = torch.stack(caps)
        cmasks = torch.stack(cmasks)

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,  # X for original code.
            "vmasks": vmasks,
            "targets": targets,
            "sequence_targets": sequence_labels,
            "video_id": vid,
            "task": task,
            "video_len": video_len,  # for later checking.
        }

    def _read_assignment(self, T, K, path):
        """
        refactored from https://github.com/DmZhukov/CrossTask/blob/master/data.py
        Howto interpret contraints on loss that is going to be minimized:
        lambd is a big number;
        self.lambd * C is a big number for all valid position (csv stores invalids)

        def forward(self, O, Y, C):
            return (Y*(self.lambd * C - self.lsm(O))).mean(dim=0).sum()

        This will load the csv file and fill-in the step col from start to end rows.
        """

        Y = np.zeros([T, K], dtype=np.uint8)
        with open(path, "r") as f:
            for line in f:
                step, start, end = line.strip().split(",")
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                step = int(step) - 1
                Y[start:end, step] = 1
        return Y

    def _read_sequence(self, path):
        sequence = []
        with open(path, "r") as f:
            for line in f:
                step, _, _ = line.strip().split(",")
                sequence.append(int(step) - 1)
        return torch.tensor(sequence)
