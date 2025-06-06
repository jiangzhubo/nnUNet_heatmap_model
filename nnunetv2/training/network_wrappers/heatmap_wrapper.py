import torch
from torch import nn


class SegmentationHeatmapWrapper(nn.Module):
    """Wraps a segmentation network to additionally predict a heatmap."""

    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network
        self.num_segmentation_outputs = getattr(base_network, "num_classes", None)
        if self.num_segmentation_outputs is None:
            raise RuntimeError("Base network must define num_classes")
        # determine conv dimensionality from first conv layer
        conv_dim = 2
        for m in base_network.modules():
            if isinstance(m, nn.Conv3d):
                conv_dim = 3
                break
            if isinstance(m, nn.Conv2d):
                conv_dim = 2
                break
        conv = nn.Conv3d if conv_dim == 3 else nn.Conv2d
        self.heatmap_head = conv(self.num_segmentation_outputs, 1, kernel_size=1)
        self.num_classes = self.num_segmentation_outputs + 1

    def forward(self, x):
        seg_out = self.base_network(x)
        if isinstance(seg_out, (list, tuple)):
            out_combined = []
            for o in seg_out:
                heat = self.heatmap_head(o)
                out_combined.append(torch.cat([o, heat], dim=1))
            return out_combined
        else:
            heat = self.heatmap_head(seg_out)
            return torch.cat([seg_out, heat], dim=1)
