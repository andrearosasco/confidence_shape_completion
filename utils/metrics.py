# import multiprocessing
import torch
from sklearn.neighbors import KDTree
import numpy as np

from configs import Config


# def chamfer_batch(predictions, meshes):
#
#     ground_truth = torch.zeros([1, 8192, 3], device=Config.General.device)
#     for i, mesh in enumerate(meshes):
#         ground_truth[i] = torch.tensor(np.array(mesh.sample_points_uniformly(8192).points)).unsqueeze(0)
#
#     d = chamfer_distance(predictions, ground_truth)
#     d2 = ChamferDistanceL1()(predictions, ground_truth)
#     return d.mean().detach().cpu()
try:
    from extensions.chamfer_dist import ChamferDistanceL1
    l1 = True
except ModuleNotFoundError:
    print('warning: using python chamfer l2')
    l1 = False


def chamfer_batch_pc(predictions, ground_truth):

    # d2 = ChamferDistanceL2()(predictions, ground_truth)
    # d3 = ChamferDistanceL1(ignore_zeros=True)(torch.rand(1, 8192, 3, device='cuda'), torch.rand(1, 8192, 3, device='cuda'))
    #
    if l1:
        d1 = ChamferDistanceL1(ignore_zeros=True)(predictions.contiguous(), ground_truth.contiguous())
    else:
        d1 = chamfer_distance(predictions.contiguous(), ground_truth.contiguous()).mean()
    # d2 = torch.tensor([0.], device='cuda')
    # for pc, gt in zip(predictions, ground_truth):
    #     d2 += ChamferDistanceL1(ignore_zeros=True)(pc.unsqueeze(0).contiguous(), gt.unsqueeze(0).contiguous())
    # d2 = d2 / predictions.shape[0]
    # for pc in predictions:
    # d1 = chamfer_distance(predictions, ground_truth).mean()

    return d1


def chamfer_distance(points1, points2, give_id=False):
    """ KD-tree based implementation of the Chamfer distance.

        Args:
            points1 (numpy array): first point set
            points2 (numpy array): second point set
            give_id (bool): whether to return the IDs of the nearest points
    """
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    if isinstance(points1, torch.Tensor) and isinstance(points2, torch.Tensor):
        points1_np = points1.detach().cpu().numpy()
        points2_np = points2.detach().cpu().numpy()
    else:
        raise ValueError('Arguments have to be both tensors')

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(np.array(idx_nn_12)).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(np.array(idx_nn_21)).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)
    # chamfer1 = torch.sqrt(torch.sqrt((points1 - points_12).pow(2)).sum(2)).mean(1)
    # chamfer2 = torch.sqrt(torch.sqrt((points2 - points_21).pow(2)).sum(2)).mean(1)


    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances