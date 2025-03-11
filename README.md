<div align="center">
  <h1><b> COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric <br> Human Activity Recognition </b></h1>
</div>

<div align="center">

### [Baiyu Chen](https://baiyuchen.work/)<sup>1,2</sup>, [Wilson Wongso](https://wilsonwongso.dev)<sup>1,2</sup>, [Zechen Li](https://scholar.google.com/citations?user=EVOzBF4AAAAJ&hl=en)<sup>1</sup>, [Yonchanok Khaokaew](https://scholar.google.com/citations?user=gk2wKhIAAAAJ&hl=en)<sup>1,2</sup>, [Hao Xue](https://www.unsw.edu.au/staff/hao-xue)<sup>1,2</sup>, and [Flora Salim](https://fsalim.github.io/)<sup>1,2</sup>

<sup>1</sup> School of Computer Science and Engineering, University of New South Wales, Sydney, Australia<br/>
<sup>2</sup> ARC Centre of Excellence for Automated Decision Making + Society

[![arXiv](https://img.shields.io/badge/arXiv-2503.07259-b31b1b.svg)](https://arxiv.org/pdf/2503.07259)
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

<p align="center">
  <img src="assets/logo.png" height="100">
</p>

## 🌟 Overview
**COMODO** is an open source framework for Cross-Modal Video-to-IMU Distillation for Efficient Egocentric Human Activity Recognition.

### 🔑 The key features of COMODO:
- ***Self-supervised Cross-modal Knowledge Transfer***: We propose COMODO, a cross-modal self-supervised distillation framework that leverages pretrained video and time-series models enabling label-free knowledge transfer from a stronger modality (video) with richer training data to a weaker modality (IMU) with limited data. 
- ***A Self-supervised and Effective Cross-modal Queuing Mechanism***:  We introduce a cross-modal FIFO queue that maintains video embeddings as a stable and diverse reference distribution for IMU feature distillation, extending the instance queue distribution learning approach from single-modality to cross-modality.
- ***Teacher-Student Model Agnostic***: COMODO supports diverse video and time-series pretrained models, enabling flexible teacher-student configurations and future integration with stronger foundation models.
- ***Cross-dataset Generalization***: We demonstrate that COMODO maintains superior performance even when evaluated on unseen datasets, and more superior than fully supervised models, highlighting its robustness and generalizability for egocentric HAR tasks.

### 📂 Data & Results
All experimental results and ablation study findings can be found in the [`/results`](./results) folder.

The [`/dataset`](./dataset) folder contains the train, val, and test splits for each dataset, along with our preprocessing scripts. Specifically, ego4d_subset_ids.txt is a subset of all available IMU-containing IDs, which we obtained by applying the official Ego4D filter from their website. This represents the complete subset of data that we can access.

## 🚀 Getting started

### Cross-modal Self-supervised Distillation
To run a Self-supervised Video-to-IMU Distillation, use the following command:

> Note: `[ ]` denotes optional parameters. 

> Currently supported pretrained models:
> - Time-series models: MOMENT, Mantis  
> - Video models: VideoMAE, TimeSformer  
> 
> Other pretrained models **can be used with minor modifications to the code**.


```bash
python train.py \
    --video_ckpt "facebook/timesformer-base-finetuned-k400" \
    --imu_ckpt "paris-noah/Mantis-8M" \
    --dataset_path "DATASET_PATH" \
    --encoded_video_path "ENCODED_VIDEO_PATH" \
    --anchor_video_path "ANCHOR_VIDEO_PATH" \
    [--queue_size QUEUE_SIZE] \
    [--student_temp STUDENT_TEMP] \
    [--teacher_temp TEACHER_TEMP] \
    [--learning_rate LR] \
    [--num_epochs EPOCH] \
    [--batch_size BS] \
    [--num_clips 0] \
    [--seed SEED] \
    [--mlp_hidden_dim MLP_HIDDEN_DIM] \
    [--mlp_output_dim MLP_OUTPUT_DIM] \
    [--reduction "concat"] \
    [--is_raw true]
```

### Unsupervised Representation Learning Evaluation
We evaluate the learned IMU representations in an unsupervised manner. See Section 3.2 in our [paper](https://arxiv.org/pdf/2503.07259). We train a Support Vector Machine (SVM) on the extracted IMU features and evaluate classification accuracy on the test set. Run the following command to start the evaluation:

```bash
python unsupervised_rep_test.py \
    --imu_ckpt "AutonLab/MOMENT-1-small" \
    --model_path "MODEL_WEIGHT_PATH" \
    --dataset_path "DATASET_PATH" \
```

## 🌍 Related Works & Baselines

There's a lot of outstanding work on time-series and human activity recognition! Here's an incomplete list. Checkout Table 1 in our [paper](https://arxiv.org/pdf/2503.07259) for IMU-based Human Activity Recognition comparisons with these studies:

- **MOMENT**: A Family of Open Time-series Foundation Models [[Paper](https://arxiv.org/pdf/2402.03885), [Code](https://github.com/moment-timeseries-foundation-model/moment), [Hugging Face](https://huggingface.co/AutonLab/MOMENT-1-small)]
- **Mantis**: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification [[Paper](https://arxiv.org/pdf/2502.15637), [Code](https://github.com/vfeofanov/mantis), [Hugging Face](https://huggingface.co/paris-noah/Mantis-8M)]
- **TimesNet**: Temporal 2D-Variation Modeling for General Time Series Analysis [[Paper](https://arxiv.org/pdf/2210.02186), [Code](https://github.com/thuml/Time-Series-Library)]
- **DLinear**: Are Transformers Effective for Time Series Forecasting? [[Paper](https://arxiv.org/pdf/2205.13504.pdf), [Code](https://github.com/thuml/Time-Series-Library)]
- **Informer**: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[Paper](https://arxiv.org/pdf/2012.07436), [Code](https://github.com/thuml/Time-Series-Library)]
- **IMU2CLIP**: Language-grounded Motion Sensor Translation with Multimodal Contrastive Learning [[Paper](https://aclanthology.org/2023.findings-emnlp.883.pdf), [Code](https://github.com/facebookresearch/imu2clip)]

## 📩 Contact

If you have any questions or suggestions, feel free to contact Baiyu (Breeze) at `breeze.chen(at)student(dot)unsw(dot)edu(dot)au`.

<img align="right" height="100px" src="assets/arc_centre.svg" style="margin-left: 10px;">
<img align="right" height="100px" src="assets/adms_logo.svg" style="margin-left: 10px;">
<img align="right" height="100px" src="assets/unsw_logo.png">