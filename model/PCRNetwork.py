import copy
from abc import ABC
from collections import defaultdict
import numpy as np
import torch

from datasets import ShapeCompletionDataset
from .Backbone import BackBone
from .Decoder import Decoder
from .ImplicitFunction import ImplicitFunction

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.geometry import PointCloud

from configs import Config

try:
    from torchmetrics import F1Score
except ImportError:
    from torchmetrics import F1 as F1Score

from torchmetrics import Accuracy, Precision, Recall, MeanMetric
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.pointclouds import sample_point_cloud_pc, voxelize_pc
import pytorch_lightning as pl
import wandb


class PCRNetwork(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)
        self.decoder = Decoder(self.sdf)  # 8192*2, 0.7, 20

        # self.apply(self._init_weights)
        for parameter in self.backbone.transformer.parameters():
            if len(parameter.size()) > 2:
                torch.nn.init.xavier_uniform_(parameter)

        m = {'accuracy': Accuracy().cuda(),
             'precision': Precision().cuda(),
             'recall': Recall().cuda(),
             'f1': F1Score().cuda(),
             'loss': MeanMetric().cuda(), }

        self.metrics = {
            'train': m,
            'val_models': {**copy.deepcopy(m), **{'jaccard': MeanMetric().cuda()}},
            'val_views': {**copy.deepcopy(m), **{'jaccard': MeanMetric().cuda()}}
        }

        self.training_set, self.valid_set_models, self.valid_set_views = None, None, None

        self.rt_setup = False

        self.cls_count = defaultdict(lambda: 0)
        self.cls_cd = defaultdict(lambda: 0)

        self.cd = []
        self.labels = []

        self.val_outputs = None

    def prepare_data(self):
        root = Config.Data.dataset_path
        split = 'data/MCD/build_datasets/train_test_dataset.json'

        self.training_set = ShapeCompletionDataset(root, split, subset='train_models_train_views')
        self.valid_set_models = ShapeCompletionDataset(root, split, subset='holdout_models_holdout_views', length=3200)

    def train_dataloader(self):
        dl = DataLoader(self.training_set,
                        batch_size=Config.Train.mb_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=Config.General.num_workers,
                        pin_memory=True)

        return dl

    def val_dataloader(self):
        # # Smaller one
        # dl1 = DataLoader(
        #     self.valid_set_views,
        #     shuffle=False,
        #     batch_size=Config.Eval.mb_size,
        #     drop_last=False,
        #     num_workers=Config.General.num_workers,
        #     pin_memory=True)

        #  Bigger one
        dl2 = DataLoader(
            self.valid_set_models,
            shuffle=False,
            batch_size=Config.Eval.mb_size,
            drop_last=False,
            num_workers=Config.General.num_workers,
            pin_memory=True)

        return dl2

    def forward(self, partial):
        decoder = Decoder(self.sdf)
        fast_weights, _ = self.backbone(partial)
        pc, prob = decoder(fast_weights)

        return pc, prob

    def configure_optimizers(self):
        optimizer = Config.Train.optimizer(self.parameters(), lr=Config.Train.lr, weight_decay=Config.Train.wd)
        return optimizer

    def on_train_epoch_start(self):
        for m in self.metrics['train'].values():
            m.reset()

    def training_step(self, batch, batch_idx):
        partial, ground_truth = batch

        fast_weights, _ = self.backbone(partial)
        # ### Adaptation
        # if Config.Train.adaptation:
        #     adaptation_steps = 10
        #
        #     fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]
        #
        #     for _ in range(adaptation_steps):
        #         optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1,
        #                                   momentum=0.9)  # the sum flatten the list of list
        #
        #         out = self.sdf(partial, fast_weights)
        #         # The loss function also computes the sigmoid
        #         loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,),
        #                                                                   device=Config.General.device))
        #
        #         loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
        #         fast_weights = optim.step()
        #         fast_weights = [[fast_weights[i].to(Config.General.device),
        #                          fast_weights[i + 1].to(Config.General.device),
        #                          fast_weights[i + 2].to(Config.General.device)]
        #                         for i in range(0, 3 * (Config.Model.depth + 2), 3)]

        samples, target = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
                                                dist=Config.Data.dist,
                                                noise_rate=Config.Data.noise_rate,
                                                tolerance=Config.Data.tolerance)

        out = self.sdf(samples, fast_weights)
        loss = F.binary_cross_entropy_with_logits(out, target.unsqueeze(-1).to(out.dtype))

        pred = torch.sigmoid(out.detach()).cpu()
        target = target.unsqueeze(-1).int()

        return {'loss': loss, 'pred': pred.detach().cpu(), 'target': target.detach().cpu()}

    @torch.no_grad()
    def training_step_end(self, output):
        pred, trgt = output['pred'], output['target']
        pred = torch.nan_to_num(pred)

        # This log the metrics on the current batch and accumulate it in the average
        metrics = self.metrics['train']

        self.log('train/accuracy', metrics['accuracy'](pred, trgt))
        self.log('train/precision', metrics['precision'](pred, trgt))
        self.log('train/recall', metrics['recall'](pred, trgt))
        self.log('train/f1', metrics['f1'](pred, trgt))

        self.log('train/loss', metrics['loss'](torch.nan_to_num(output['loss'].detach().cpu(), nan=1)))

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.random_batch = np.random.randint(1, int(len(self.valid_set_models) / Config.Eval.mb_size))
        self.val_outputs = {}

        for m in self.metrics['val_models'].values():
            m.reset()
        for m in self.metrics['val_views'].values():
            m.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        partial, ground_truth = batch

        if 0 == batch_idx or batch_idx == self.random_batch:
            samples2, occupancy2 = sample_point_cloud_pc(ground_truth,
                                                         n_points=Config.Data.implicit_input_dimension,
                                                         dist=Config.Data.dist,
                                                         noise_rate=Config.Data.noise_rate,
                                                         tolerance=Config.Data.tolerance)
            fast_weights, _ = self.backbone(partial)
            pc1, _ = self.decoder(fast_weights)
            out2 = torch.sigmoid(self.sdf(samples2, fast_weights))

            key = 'fixed' if batch_idx == 0 else 'random'
            self.val_outputs[key] = {'pc1': pc1, 'out2': out2.detach().squeeze(2).cpu(),
                                     'target2': occupancy2.detach().cpu(), 'samples2': samples2.detach().cpu(),
                                     'partial': partial.detach().cpu(), 'ground_truth': ground_truth,
                                     'batch_idx': batch_idx}

        # The sampling with "sample_point_cloud" simulate the sampling used during training
        # This is useful as we always get ground truths sampling on the meshes but it doesn't reflect
        #   how the algorithm will work after deployment

        samples2, occupancy2 = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
                                                     dist=Config.Data.dist,
                                                     noise_rate=Config.Data.noise_rate, tolerance=Config.Data.tolerance)

        ############# INFERENCE #############
        fast_weights, _ = self.backbone(partial)

        ### Adaptation
        # if Config.Train.adaptation:
        #     with torch.enable_grad():
        #         adaptation_steps = 10
        #
        #         fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]
        #
        #         for _ in range(adaptation_steps):
        #             optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1,
        #                                       momentum=0.9)  # the sum flatten the list of list
        #
        #             out = self.sdf(partial, fast_weights)
        #             loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,),
        #                                                                       device=Config.General.device))
        #
        #             loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
        #             fast_weights = optim.step()
        #             fast_weights = [[fast_weights[i], fast_weights[i + 1], fast_weights[i + 2]] for i in
        #                             range(0, 3 * (Config.Model.depth + 2), 3)]
        from dgl.geometry import farthest_point_sampler
        pc1, _ = self.decoder(fast_weights)
        point_idx = farthest_point_sampler(pc1, 8192 * 2)
        pc1 = pc1[torch.arange(point_idx.shape[0]).unsqueeze(-1), point_idx]

        # fp_idxs = fp_sampling(pc1[0:2], 1024)
        # fp_pc1 = pc1[0:2][torch.arange(fp_idxs.shape[0]).unsqueeze(-1), fp_idxs.long(), :]
        # vx_pc1 = voxel_downsample(pc1, 0.005)
        #
        # import mcubes
        # grid1 = voxelize_pc(pc1, 0.005)
        # vertices, triangles = mcubes.marching_cubes(grid1[0].cpu().numpy(), 0.5)
        # mesh = o3d.geometry.TriangleMesh(triangles=Vector3iVector(triangles), vertices=Vector3dVector(vertices))
        #
        out2 = self.sdf(samples2, fast_weights)
        pred2 = torch.sigmoid(out2)

        out, pred, trgt = out2.detach().squeeze(2), pred2.detach().squeeze(2), \
                          occupancy2.detach().int()

        loss = F.binary_cross_entropy_with_logits(out, trgt.to(out.dtype))

        pred = torch.nan_to_num(pred)

        metrics = self.metrics['val_models']

        metrics['accuracy'](pred, trgt), metrics['precision'](pred, trgt)
        metrics['recall'](pred, trgt), metrics['f1'](pred, trgt)
        metrics['loss'](loss)
        grid1 = voxelize_pc(pc1, 0.025)
        grid2 = voxelize_pc(ground_truth, 0.025)
        jaccard = torch.sum(grid1 * grid2, dim=[1, 2, 3]) / torch.sum((grid1 + grid2) != 0, dim=[1, 2, 3])

        metrics['jaccard'](jaccard)

    @torch.no_grad()
    def validation_epoch_end(self, output):
        # self.log(f'val_models/jaccard', self.metrics['val_models']['jaccard'].compute())
        for k, m in self.metrics['val_models'].items():
            self.log(f'val_models/{k}', m.compute())

        if self.val_outputs is None:
            return

        for name, batch in self.val_outputs.items():

            out, trgt, samples = batch['out2'][-1], batch['target2'][-1], batch['samples2'][-1]

            pred = out > 0.5

            # all positive predictions with labels for true positive and false positives
            colors = torch.zeros_like(samples, device='cpu')
            colors[trgt.bool().squeeze()] = torch.tensor([0, 255., 0])
            colors[~trgt.bool().squeeze()] = torch.tensor([255., 0, 0])

            precision_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            precision_pc = precision_pc[pred.squeeze()]

            # all true points with labels for true positive and false negatives
            colors = torch.zeros_like(samples, device='cpu')
            colors[pred.squeeze()] = torch.tensor([0, 255., 0])
            colors[~ pred.squeeze()] = torch.tensor([255., 0, 0])

            recall_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            recall_pc = recall_pc[(trgt == 1.).squeeze()]

            complete = batch['ground_truth'][-1].detach().cpu().numpy()

            partial = torch.cat(
                [batch['partial'][-1], torch.tensor([[255., 165., 0.]]).tile(batch['partial'][-1].shape[0], 1)],
                dim=-1).detach().cpu().numpy()

            reconstruction = batch['pc1'][-1]
            idxs = (reconstruction[..., 0] > 0.5) + (reconstruction[..., 0] < -0.5) + (reconstruction[..., 1] > 0.5) + (
                    reconstruction[..., 1] < -0.5) + (
                           reconstruction[..., 2] > 0.5) + (reconstruction[..., 2] < -0.5)
            reconstruction = reconstruction[~idxs]
            reconstruction = torch.cat(
                [reconstruction.detach().cpu(), torch.tensor([[0., 0., 255.]]).tile(reconstruction.shape[0], 1)],
                dim=-1).numpy()

            if Config.Eval.wandb:
                experiment = self.trainer.logger.experiment
                if isinstance(experiment, list):
                    experiment = experiment[0]

                experiment.log(
                    {f'{name}_precision_pc': wandb.Object3D({"points": precision_pc, 'type': 'lidar/beta'})})
                experiment.log(
                    {f'{name}_recall_pc': wandb.Object3D({"points": recall_pc, 'type': 'lidar/beta'})})
                experiment.log(
                    {f'{name}_partial_pc': wandb.Object3D({"points": partial, 'type': 'lidar/beta'})})
                experiment.log(
                    {f'{name}_complete_pc': wandb.Object3D({"points": complete, 'type': 'lidar/beta'})})
                experiment.log(
                    {f'{name}_reconstruction': wandb.Object3D(
                        {"points": np.concatenate([reconstruction, partial], axis=0), 'type': 'lidar/beta'})})
