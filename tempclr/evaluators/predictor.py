# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import json
import numpy as np
import torch
import pickle
import math

from tqdm import tqdm


class Predictor(object):
    """this base class is used to save predictions to disk
        (and being called by a evaluator later).
        Predictor has minimum support of single gpu prediction.
    """
    def __init__(self, config):
        self.pred_dir = None  # on-the-fly eval does not save the results.
        if hasattr(config, "eval") and config.eval is not None:
            self.pred_dir = config.eval.save_path
            os.makedirs(self.pred_dir, exist_ok=True)

    def __call__(self, outputs):
        """extract the prediction and save it."""
        raise NotImplementedError

    def predict_loop(self, model, eval_dataloader, output_file=None):
        """on-the-fly prediction on a single gpu."""
        self.full_scores = []
        model.eval()
        model = model.to(0)
        with torch.no_grad():
            for data in eval_dataloader:
                data = self.to_ctx(data)
                outputs = model(**data)
                outputs.update(data)
                self(outputs)
        return self.finalize(output_file)

    def finalize(self, output_file):
        pass

    def to_ctx(self, data, ctx=0, dtype=None):
        if isinstance(data, dict):
            for key in data:
                if torch.is_tensor(data[key]):
                    if dtype is not None and data[key].dtype == torch.float32:
                        data[key] = data[key].to(dtype)
                    data[key] = data[key].to(ctx)
            return data
        else:
            raise ValueError("non-dict type of batch is not supported yet.")


class RetrievalPredictor(Predictor):
    """generated `pooled_video` and `pooled_text`."""
    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.dataset.bert_name)

    def predict_loop(
        self,
        model,
        eval_dataloader,
        output_file="retrieval.npy"
    ):
        """on-the-fly prediction on a single gpu."""
        full_scores = []
        texts = []
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            for data in eval_dataloader:
                # convert to dict.
                if not isinstance(data, dict):
                    data = {
                        "caps": data[0],
                        "cmasks": data[1],
                        "vfeats": data[2],
                        "vmasks": data[3],
                        "video_id": data[4]
                    }
                data = self.to_ctx(data)
                outputs = model(**data)
                outputs.update(data)
                outputs['video_id'] = list(data['video_id'][0])

                self(outputs, full_scores)
                for _cap in data["caps"]:
                    texts.append(
                        self.tokenizer.decode(_cap, skip_special_tokens=True)
                    )

        return self.finalize(full_scores, texts, output_file)

    def __call__(self, sample, full_scores):
        scores = self._get_pooled_outputs(sample)
        self._append_scores(scores, full_scores)

    def finalize(self, full_scores, texts, output_file=None):
        outputs, video_ids = self._aggregate_scores(full_scores)
        if output_file is not None:
            np.save(os.path.join(self.pred_dir, output_file + ".npy"), outputs)
        return {"outputs": outputs, "texts": texts, "video_ids": video_ids}

    def _get_pooled_outputs(self, outputs):
        if "pooled_video" in outputs:
            return outputs["pooled_video"], outputs["pooled_text"], outputs['video_id']
        else:
            raise ValueError("unknown format of outputs.")

    def _append_scores(self, scores, full_scores):
        if len(full_scores) == 0:
            full_scores.append([])
            full_scores.append([])
            full_scores.append([])

        full_scores[0].append(scores[0].cpu().detach().numpy())
        full_scores[1].append(scores[1].cpu().detach().numpy())
        full_scores[2] += scores[2]

    def _aggregate_scores(self, scores):
        video_hidden = np.concatenate(scores[0], axis=0)
        text_hidden = np.concatenate(scores[1], axis=0)

        video_ids = scores[2]
        self.full_scores = []

        return np.matmul(text_hidden, video_hidden.T), video_ids


class CrossTaskPredictor(Predictor):
    """
    CrossTaskPredictor needs to compute the average of logits
    for overlapped sliding-window.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.max_video_len = config.dataset.max_video_len
        self.sliding_window = config.dataset.sliding_window
        self.sliding_window_size = config.dataset.sliding_window_size
        self.annotation_path = config.dataset.annotation_path

    def predict_loop(self, model, eval_dataloader, output_file="result.pkl"):
        """refactored from line 144:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py
        """
        ctx = 0
        model.eval()
        model = model.to(ctx)
        # this is not a loss but just compute neg_log_prob.
        Y_pred = {}
        Y_true = {}
        with torch.no_grad():
            for batch in eval_dataloader:
                self(batch, model, Y_pred, Y_true)
        return self.finalize(Y_pred, Y_true, output_file)

    def __call__(self, sample, model, Y_pred, Y_true):
        # please install dp from `https://github.com/DmZhukov/CrossTask`
        from dp import dp
        vid, task = sample['video_id'][0], sample['task'][0]
        sample = self.to_ctx(sample)
        # compute the average logits over sliding windows.
        output = model(**sample)
        batch_logits = output["logits"].cpu()

        video_len = sample["video_len"][0]

        # the following version is slow.
        logits = torch.zeros((video_len, batch_logits.size(1)))
        logits_counts = torch.zeros((video_len, 1), dtype=torch.long)
        # use the same loop as aligner to recover.
        batch_logit_idx = 0
        for window_start in range(0, video_len, self.sliding_window):
            video_end = min(video_len - window_start, self.sliding_window_size)
            logits[window_start: window_start + video_end] += batch_logits[
                batch_logit_idx: batch_logit_idx + video_end]
            batch_logit_idx += video_end
            logits_counts[window_start: window_start + video_end] += torch.ones((video_end, 1), dtype=torch.long)

            if (video_len - window_start) <= self.sliding_window_size:
                break

        logits /= logits_counts
        assert logits.size() == (video_len, batch_logits.size(1)), "{}, {}".format(logits.size(), video_len)

        O = self.lsm(logits)
        y = np.zeros(O.size(), dtype=np.float32)
        dp(y, -O.detach().cpu().numpy())
        if task not in Y_pred:
            Y_pred[task] = {}
        Y_pred[task][vid] = y
        annot_path = os.path.join(
            self.annotation_path, task+'_'+vid+'.csv')
        if os.path.exists(annot_path):
            if task not in Y_true:
                Y_true[task] = {}
            Y_true[task][vid] = self._read_assignment(
                *y.shape, annot_path)

    def finalize(self, Y_pred, Y_true, output_file=None):
        if output_file is not None:
            with open(
                    os.path.join(self.pred_dir, output_file + ".pkl"),
                    "wb") as fw:
                pickle.dump(
                    {"Y_pred": Y_pred, "Y_true": Y_true}, fw,
                    protocol=pickle.HIGHEST_PROTOCOL)
        return {"outputs": Y_pred, "targets": Y_true}

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
        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                step = int(step) - 1
                Y[start:end, step] = 1
        return Y

