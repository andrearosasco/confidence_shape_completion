# As soon as pytorch supports cdist onnx, throw this away and use that
import torch


def minimum(x1, x2):
    return torch.where(x2 < x1, x2, x1)


def cdists(a, b):
    """ Custom cdists function for ONNX export since neither cdists nor
    linalg.norm is currently support by the current PyTorch version 1.10.
    """
    a = a.unsqueeze(2)  # add columns dimension and repeat along it
    a = a.repeat(1, 1, b.shape[1], 1)
    b = b.unsqueeze(1)  # add rows dimension and repeat along it
    b = b.repeat(1, a.shape[1], 1, 1)

    res = (a - b).pow(2).sum(-1).sqrt()
    return res


def fp_sampling(points, num: int):
    batch_size = points.shape[0]
    # use onnx.cdists just to export to onnx, otherwise use torch.cdist
    D = cdists(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    res = torch.zeros((batch_size, 1), dtype=torch.int32, device=points.device)
    ds = D[:, 0, :]
    for i in range(1, num):
        idx = ds.max(dim=1)[1]
        res = torch.cat([res, idx.unsqueeze(1).to(torch.int32)], dim=1)
        ds = minimum(ds, D[torch.arange(batch_size), idx, :])

    return res
