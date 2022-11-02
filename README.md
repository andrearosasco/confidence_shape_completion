# Towards Confidence-guided Shape Completion for Robotic Applications
Code for [Towards Confidence-guided Shape Completion for Robotic Applications](https://arxiv.org/abs/2209.04300)

## Requirements
```
open3d=0.14.1
pytorch-lightning=1.6.4
scipy
timm
torchmetrics
wandb
```
## Run the code
To measure the jaccard similarity on the datasets prepare the dataset following the instruction on `data/MCD/README.md` and simply run `eval/measure_jaccard.py`
