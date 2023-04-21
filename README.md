# TempCLR: Temporal Alignment Representation with Contrastive Learning [ICLR 2023]
Codebase for ICLR 2023 submission [TempCLR: Temporal Alignment Representation with Contrastive Learning](https://arxiv.org/abs/2212.13738)

Our work is developed based on the [MMPT toolkit](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT). Great thanks to the VideoCLIP team for open-sourcing the toolkit!

## Updates
- (April 2023) Added the evaluation pipeline for background kept full video retrieval on youcook2
- (March 2023) The checkpoints pre-trained using DTW and OTAM has been [published](https://drive.google.com/drive/folders/1aD6l8yp0dsPpRKbmg3a_CBiK6UVN8GrR?usp=sharing).

## Installation
We use fairseq as the main trainer:  
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy.
```

Then install the TempCLR package:
```
git clone https://github.com/yyuncong/TempCLR
cd TempCLR
pip install -e .
```

Install CrossTask if need to run experiments on Action Step Localization:
```
git clone https://github.com/DmZhukov/CrossTask
cd CrossTask
pip install -e .
```

The code is developed under Python=3.8.8.


## Usage
#### Download Checkpoints
We use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction. Please place the models as `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`.

Download VideoCLIP checkpoint `https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt` to `runs/retri/pretrained/videoclip`.

Based on the best VideoCLIP checkpoint, we finetuned [3 checkpoints](https://drive.google.com/drive/folders/1aD6l8yp0dsPpRKbmg3a_CBiK6UVN8GrR?usp=sharing) on a small subset (7.5%) of HowTo100M using OTAM and DTW metrics. The majority of the results we have reported are from the layernorm checkpoints.

`checkpoint_layernorm_OTAM`: Only layernorm in the original model was finetuned and we used OTAM as the metric for fine-tuning.

`checkpoint_layernorm_DTW`: Only layernorm in the original model was finetuned and we used DTW as the metric for fine-tuning.

`checkpoint_allparameters_DTW`: All parameters in the original model was finetuned and we used DTW as the metric for fine-tuning. (This is only for zero-shot action step localization. *Performance drops for any other tasks*)

#### Data Preparation
See [dataset](https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/DATASET.md) from the MMPT toolkit for each dataset.

Feature extraction on Howto100M requires large computational resourse. Official features from [Howto100M](https://www.di.ens.fr/willow/research/howto100m/) could be a substitute for quickstart, but the model may have performance drop comparing to data reported in our paper (The official features are different from the features that our baseline VideoCLIP model pre-trained on). 

#### Global Config for Training Pipeline
We wrap all cmds into `locallaunch.py` and `tempclr_cli/localjob.py`. You can check concrete cmds by `--dryrun` and then drop it for actual run.  

Each training or evaluation process will be configed by a concrete config file. We save all complex arguments into the concrete config file for reproducibility. For example, run training/evaluation on crosstask,
```
python locallaunch.py configs/test_crosstask_zs.yaml --jobtype local_predict  # zero-shot evaluation.
python locallaunch.py configs/crosstask.yaml --jobtype local_single --dryrun  # fine-tuning: use --dryrun to check cmds and drop it to make an actual run.
python locallaunch.py configs/test_crosstask.yaml --jobtype local_predict  # testing on fine-tuned model.
```

Pretraining can be run as:  
```
python locallaunch.py configs/how2.yaml --jobtype local_single --dryrun
```

### Performance-tuned Components
To speed up pre-training, this toolkit uses sharded features stored in mmaped numpy, backed by `ShardedTensor` in `tempclr/utils/shardedtensor.py` . This reduces the loads of IO for multi-GPU training without loading all features for a video into the memory each time and `ShardedTensor` ensure features are stored in continuous disk space for near random access. This is used for both How2 video features and texts.


## Citation
If you find this codebase helpful for your work, please cite our paper:
```BibTeX
@inproceedings{yang2023tempclr,
  title={TempCLR: Temporal Alignment Representation with Contrastive Learning},
  author={Yang*, Yuncong and Ma*, Jiawei and Huang, Shiyuan and Chen, Long and Lin, Xudong and Han, Guangxing and Chang, Shih-Fu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## Issues
We will keep update the codebase. Please report issues to the Issues section of this repo. The Issues section of the [MMPT toolkit](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT) could be helpful as well.

## Copyright
The majority of TempCLR is under MIT License, however portions of the project are available under separate license terms: Evaluation Codes/Models: Howto100M and HuggingFace Transformers are licensed under the Apache2.0 license; CrossTask is licensed under the BSD-3
