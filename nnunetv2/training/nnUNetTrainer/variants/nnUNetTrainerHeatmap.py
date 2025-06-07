from __future__ import annotations
from typing import List

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_wrappers.heatmap_wrapper import (
    SegmentationHeatmapWrapper,
)


class nnUNetTrainerHeatmap(nnUNetTrainer):
    """Trainer that adds a heatmap regression head to the network."""

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: List[str],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base = super().build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return SegmentationHeatmapWrapper(base)

    def _build_loss(self):
        seg_loss = super()._build_loss()
        mse = nn.MSELoss()
        num_seg = self.label_manager.num_segmentation_heads

        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.seg_loss = seg_loss
                self.mse = mse

            def forward(self, outputs, target_tuple):
                target, heatmap = target_tuple
                if isinstance(outputs, (list, tuple)):
                    seg_out = [o[:, :num_seg] for o in outputs]
                    heat_out = [o[:, num_seg:] for o in outputs]
                    seg_l = self.seg_loss(seg_out, target)
                    heat_l = sum(self.mse(h, heatmap) for h in heat_out) / len(heat_out)
                else:
                    seg_l = self.seg_loss(outputs[:, :num_seg], target)
                    heat_l = self.mse(outputs[:, num_seg:], heatmap)
                return seg_l + heat_l

        return CombinedLoss()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        seg_target = batch["target"]
        heatmap_target = batch["heatmap"]
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
        heatmap_target = heatmap_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            output = self.network(data)
            loss = self.loss(output, (seg_target, heatmap_target))

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        seg_target = batch["target"]
        heatmap_target = batch["heatmap"]
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
        heatmap_target = heatmap_target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            output = self.network(data)
            loss = self.loss(output, (seg_target, heatmap_target))

        if self.enable_deep_supervision:
            output = output[0]
            seg_target = seg_target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            pred_onehot = (torch.sigmoid(output[:, :num_seg]) > 0.5).long()
        else:
            seg = output[:, :num_seg].argmax(1)[:, None]
            pred_onehot = torch.zeros_like(output[:, :num_seg])
            pred_onehot.scatter_(1, seg, 1)
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (seg_target != self.label_manager.ignore_label).float()
                seg_target[seg_target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - seg_target[:, -1:]
                seg_target = seg_target[:, :-1]
        else:
            mask = None
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, seg_target, axes=axes, mask=mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy(),
            "fp_hard": fp.detach().cpu().numpy(),
            "fn_hard": fn.detach().cpu().numpy(),
        }
