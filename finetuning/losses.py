import torch
import torch.nn.functional as F


def identity_loss(output_embedding, source_embedding):
    output_embedding = F.normalize(output_embedding, dim=1)
    source_embedding = F.normalize(source_embedding, dim=1)
    return 1.0 - (output_embedding * source_embedding).sum(dim=1).mean()


def baseline_preservation_loss(output, baseline):
    return F.l1_loss(output, baseline)


def edge_map(image):
    gray = image.mean(dim=1, keepdim=True)
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


def edge_preservation_loss(output, target):
    return F.l1_loss(edge_map(output), edge_map(target))


def self_reconstruction_loss(output, target):
    return F.l1_loss(output, target)
