"""Point-set abstraction (PointNet++ style patchify), vendored from
`models/pointnet_util.py` (repo root) - trimmed to only what Point-BERT's
patchify step (`PointNetSetAbstraction` with `group_all=False`) actually uses,
so this tutorial folder has no import-time dependency on the rest of the repo.

Pure PyTorch, no extra dependencies: the original file calls out to the `dgl`
package for farthest-point sampling, but `dgl` pulls in a large, version-fragile
dependency chain (it broke on at least one environment this notebook was tested
in) just for one function. We use the plain PyTorch farthest-point-sampling loop
instead - it's the same standard algorithm, just not GPU-kernel-optimized, which
doesn't matter at this tutorial's scale (batches of ~16, ~1024 points each).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Pairwise squared Euclidean distance. src: [B,N,C], dst: [B,M,C] -> [B,N,M]."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """points: [B,N,C], idx: [B,S] -> [B,S,C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Iterative farthest-point sampling (pure PyTorch, no external deps).
    xyz: [B,N,3] -> sampled point indices [B,npoint]."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball-query neighbor indices. xyz: [B,N,3], new_xyz: [B,S,3] -> [B,S,nsample]."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[..., :1].repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """Farthest-point-sample `npoint` centroids, then ball-query + group their neighbors.
    xyz: [B,N,3], points: [B,N,D] or None.
    Returns new_xyz [B,npoint,3], new_points [B,npoint,nsample,3(+D)]."""
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Patchify a point cloud into `npoint` local patches, each summarized by a
    small shared MLP + max-pool (the standard PointNet++ set-abstraction layer).
    Used by Point-BERT to turn N raw points into `npoint` patch tokens.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        assert not group_all, "group_all=True is unused by Point-BERT and not implemented here"
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """xyz: [B,C,N], points: [B,D,N] -> new_xyz [B,C,S], new_points [B,D',S]."""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
