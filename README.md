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
### Checkpoints and Splits
If you have git lfs install you should automatically get the model weights and the dataset splits when you clone the repository. If you don't, you can get them running
```
wget https://github.com/andrearosasco/confidence_shape_completion/blob/main/checkpoints/model.ckpt
wget https://github.com/andrearosasco/confidence_shape_completion/raw/main/data/mcd/splits/train_test_dataset.json
```

## Run the code
To measure the jaccard similarity on the datasets prepare the dataset following the instruction on `data/MCD/README.md` and simply run `eval/measure_jaccard.py`
To train the network run `train.py`


