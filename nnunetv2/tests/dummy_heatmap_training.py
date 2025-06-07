import os
import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.network_wrappers.heatmap_wrapper import SegmentationHeatmapWrapper
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 2, 3, padding=1)
        # required by SegmentationHeatmapWrapper
        self.num_classes = 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)


def get_loader(folder):
    dataset = nnUNetDatasetNumpy(folder)
    labels = {"background": 0, "object": 1}
    lm = LabelManager(labels, None)
    dl = nnUNetDataLoader(
        dataset,
        batch_size=2,
        patch_size=(32, 32),
        final_patch_size=(32, 32),
        label_manager=lm,
    )
    return dl


def train_one_epoch(net, loader, device):
    net.train()
    optim = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(5):
        batch = next(loader)
        data = torch.as_tensor(batch["data"]).to(device)
        target = torch.as_tensor(batch["target"]).squeeze(1).long().to(device)
        heat = torch.as_tensor(batch["heatmap"]).to(device)
        out = net(data)
        seg_out = out[:, :2]
        heat_out = out[:, 2].unsqueeze(1)
        loss = F.cross_entropy(seg_out, target) + F.mse_loss(heat_out, heat)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return loss.item()


def validate(net, loader, device):
    net.eval()
    with torch.no_grad():
        batch = next(loader)
        data = torch.as_tensor(batch["data"]).to(device)
        target = torch.as_tensor(batch["target"]).squeeze(1).long().to(device)
        heat = torch.as_tensor(batch["heatmap"]).to(device)
        out = net(data)
        seg_out = out[:, :2]
        heat_out = out[:, 2].unsqueeze(1)
        loss = F.cross_entropy(seg_out, target) + F.mse_loss(heat_out, heat)
    return loss.item()


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    dataset_folder = os.path.join(here, "dummy_heatmap_dataset")
    if not os.path.isdir(dataset_folder):
        from generate_dummy_heatmap_dataset import generate_dataset

        generate_dataset(dataset_folder)
    loader = get_loader(dataset_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SegmentationHeatmapWrapper(TinyNet()).to(device)
    train_loss = train_one_epoch(net, loader, device)
    val_loss = validate(net, loader, device)
    print(f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
