import copy

import numpy as np
import torch


def create_cube(num_points):
    side = num_points // 6

    p = np.random.rand(side, 3) - 0.5
    neg = np.full(side, -0.5)
    pos = np.full(side, 0.5)
    cube = np.concatenate([np.stack([neg, p[:, 1], p[:, 2]]),
                           np.stack([pos, p[:, 1], p[:, 2]]),
                           np.stack([p[:, 0], neg, p[:, 2]]),
                           np.stack([p[:, 0], pos, p[:, 2]]),
                           np.stack([p[:, 0], p[:, 1], neg]),
                           np.stack([p[:, 0], p[:, 1], pos])], axis=1).T

    return cube


@torch.no_grad()
def sample_point_cloud_pc(pc, n_points=8192, dist=None, noise_rate=0.1, tolerance=0.01, seed=1234):
    """
    http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    Produces input for implicit function
    :param mesh: Open3D mesh
    :param noise_rate: rate of gaussian noise added to the point sampled from the mesh
    :param percentage_sampled: percentage of point that must be sampled uniform
    :param total: total number of points that must be returned
    :param tollerance: maximum distance from mesh for a point to be considered 1.
    :param mode: str, one in ["unsigned", "signed", "occupancy"]
    :return: points (N, 3), occupancies (N,)
    """
    assert sum(dist) == 1.0

    n_uniform = int(n_points * dist[0])
    n_noise = int(n_points * dist[1])
    n_surface = n_points - (n_uniform + n_noise)

    points_uniform = torch.rand(pc.shape[0], n_uniform, 3, device=pc.device) - 0.5

    idx = torch.tensor(np.random.choice(pc.shape[1], n_noise, replace=True if pc.shape[1] < n_noise else False),
                       device=pc.device, dtype=torch.long)
    points_noisy = pc[:, idx, :] + torch.normal(0, noise_rate, (pc.shape[0], n_noise, 3), device=pc.device)

    idx = torch.tensor(np.random.choice(pc.shape[1], n_surface, replace=True if pc.shape[1] < n_noise else False),
                       device=pc.device, dtype=torch.long)
    points_surface = copy.deepcopy(pc[:, idx, :])

    points = torch.cat([points_uniform, points_noisy, points_surface], dim=1)

    points = points.clip(min=-0.5 + 1e-6, max=0.5 - 1e-6)
    if tolerance > 0:
        labels = check_occupancy(pc, points, tolerance)  # 0.002

    elif tolerance == 0:
        labels = [False] * (n_uniform + n_noise) + [True] * n_surface
        labels = torch.tensor(labels, dtype=torch.float, device=pc.device).tile(pc.shape[0], 1)

    return points, labels


def voxelize_pc(pc, voxel_size):
    side = int(1 / voxel_size)

    ref_idxs = ((pc + 0.5) / voxel_size).long()
    ref_grid = torch.zeros([pc.shape[0], side, side, side], dtype=torch.bool, device=pc.device)

    ref_idxs = ref_idxs.clip(min=0, max=ref_grid.shape[1] - 1)
    ref_grid[
        torch.arange(pc.shape[0]).reshape(-1, 1), ref_idxs[..., 0], ref_idxs[..., 1], ref_idxs[..., 2]] = True

    return ref_grid


def check_occupancy(reference, pc, voxel_size):
    side = int(1 / voxel_size)

    ref_idxs = ((reference + 0.5) / voxel_size).long()
    ref_grid = torch.zeros([reference.shape[0], side, side, side], dtype=torch.bool, device=pc.device)

    ref_idxs = ref_idxs.clip(min=0, max=ref_grid.shape[1] - 1)
    ref_grid[
        torch.arange(reference.shape[0]).reshape(-1, 1), ref_idxs[..., 0], ref_idxs[..., 1], ref_idxs[..., 2]] = True

    pc_idxs = ((pc + 0.5) / voxel_size).long()
    pc_idxs = pc_idxs.clip(min=0, max=ref_grid.shape[1] - 1)

    res = ref_grid[torch.arange(reference.shape[0]).reshape(-1, 1), pc_idxs[..., 0], pc_idxs[..., 1], pc_idxs[..., 2]]

    return res
