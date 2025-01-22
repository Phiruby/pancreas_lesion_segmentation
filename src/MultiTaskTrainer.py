import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np

class ClassificationHead(nn.Module):
    """
    A simple classification head for multi-task learning.
    """
    def __init__(self, input_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global average pooling for 3D data
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1)  # Flatten after pooling
        x = self.fc(x)
        return x

class nnUNetMultiTaskTrainer(nnUNetTrainer):
    """
    Custom trainer for multi-task learning: segmentation + classification.
    """
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, 
configuration, fold, dataset_json, unpack_dataset, device)
        self.num_classes_classification = 3  # Set the number of classes for classification
        self.classification_head = None

    def initialize(self):
        """
        Extend the base network with a classification head.
        """
        super().initialize()  # Initialize the base network
        encoder_output_channels = self.network.encoder.output_channels[-1]
        self.classification_head = ClassificationHead(encoder_output_channels, self.num_classes_classification).to(self.device)

    def _build_loss(self):
        """
        Build combined loss for segmentation and classification.
        """
        if self.label_manager.has_regions:
            seg_loss = DC_and_BCE_loss({},
                                       {'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                       use_ignore_label=self.label_manager.ignore_label is not None,
                                       dice_class=MemoryEfficientSoftDiceLoss)
        else:
            seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                      ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        seg_loss = DeepSupervisionWrapper(seg_loss, weights)

        return seg_loss 

    def train_step(self, data_dict):
        """
        Perform a training step with multi-task loss.
        """
        data = data_dict['data']  # Input images
        target = {
            'seg': data_dict['seg'],  # Segmentation labels
            'classification': data_dict['classification']  # Classification labels
        }

        # Forward pass
        output = self.network(data)
        total_loss, seg_loss, cls_loss = self.loss(output, target)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "seg_loss": seg_loss.item(),
            "cls_loss": cls_loss.item(),
        }

    def validation_step(self, data_dict):
        """
        Perform validation step for multi-task learning.
        """
        data = data_dict['data']
        target = {
            'seg': data_dict['seg'],
            'classification': data_dict['classification']
        }

        # Forward pass
        output = self.network(data)
        total_loss, seg_loss, cls_loss = self.loss(output, target)

        return {
            "total_loss": total_loss.item(),
            "seg_loss": seg_loss.item(),
            "cls_loss": cls_loss.item(),
        }
