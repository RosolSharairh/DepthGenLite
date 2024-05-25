
# DepthGen Lite

<img src='assets/teaser.png'/>

**Efficient Inpainting with Dual-Gen and Depthwise Convolutions**<br>

Masa Abdallah, Rosol Sharairh<br>

## Introduction

__Generator.__ Image inpainting is cast into two subtasks, _i.e._, structure-constrained texture synthesis (left, blue) and texture-guided structure reconstruction (right, red), and the two parallel-coupled streams borrow encoded deep features from each other. Depthwise convolution is used to minimize complexity. The DGDC module and CFA module are stacked at the end of the generator to further refine the results.

__Discriminator.__ The texture branch estimates the generated texture, while the structure branch guides structure reconstruction.

<img src='assets/framework.png'/>

## Prerequisites

- Python >= 3.8
- PyTorch >= 2.2.2+cu121
- NVIDIA GPU + CUDA cuDNN 11.2

## Getting Started

### Installation

- Clone this repo:

```
git clone https://github.com/MasaAbdallah/DepthGen-Lite.git
cd DepthGen-Lite
```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)
- Install python requirements:

```
pip install -r requirements.txt
```

### Datasets

**Image Dataset.** We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets, which is widely adopted in the literature. 

**Mask Dataset.** Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

### Training

Analogous to PConv by [_Liu et.al_](https://arxiv.org/abs/1804.07723), initial training followed by finetuning are performed. 

```
python train.py \
  --image_root [path to image directory] \
  --mask_root [path mask directory]

python train.py \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --pre_trained [path to checkpoints] \
  --finetune True
```

__Distributed training support.__ You can train model in distributed settings.

```
python -m torch.distributed.launch --nproc_per_node=N_GPU train.py
```

### Testing

To test the model, you run the following code.

```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```
```