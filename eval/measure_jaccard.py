import torch
from dgl.geometry import farthest_point_sampler
from torch.utils.data import DataLoader
from tqdm import trange

from configs import Config
import os

from datasets import ShapeCompletionDataset
from utils.pointclouds import voxelize_pc

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from model import PCRNetwork as Model
from torchmetrics import MeanMetric


def main():
    ckpt_path = f'checkpoints/model.ckpt'

    model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)
    model.cuda()
    model.eval()

    ds = ShapeCompletionDataset(Config.Data.dataset_path, f'{Config.Data.dataset_path}/splits/train_test_dataset.json',
                         subset='holdout_models_holdout_views')
    dl = DataLoader(
        ds,
        shuffle=False,
        batch_size=Config.Eval.mb_size,
        drop_last=False,
        num_workers=Config.General.num_workers,
        pin_memory=True)

    jaccard = MeanMetric().cuda()
    t = trange(len(dl))
    for i, data in zip(t, dl):
        partial, ground_truth = data
        partial, ground_truth = partial.cuda(), ground_truth.cuda()

        aux, _ = model(partial)
        point_idx = farthest_point_sampler(aux, 8192 * 2)
        reconstruction = aux[torch.arange(point_idx.shape[0]).unsqueeze(-1), point_idx]

        grid1 = voxelize_pc(reconstruction, 0.025)
        grid2 = voxelize_pc(ground_truth, 0.025)

        jaccard(torch.sum(grid1 * grid2, dim=[1, 2, 3]) / torch.sum((grid1 + grid2) != 0, dim=[1, 2, 3]))
        t.set_postfix(jaccard=jaccard.compute())


if __name__ == '__main__':
    main()
