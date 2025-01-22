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
        out_channels1 = input_channels * 2
        kernel_size = (3, 4, 3)
        stride = 2
        padding = (
            (kernel_size[0] - 1) // 2,
            (kernel_size[1] - 1) // 2,
            (kernel_size[2] - 1) // 2
        )
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=out_channels1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.fc = nn.Linear(input_channels, num_classes)

        self.final_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x input is of size torch.Size([3, 320, 4, 4, 6])
        # need to padd due to uneven sizing
        x = self.global_pool(x).view(x.size(0), -1)  # Flatten after pooling
        x = self.fc(x)
        # should produce something like
        # tensor([[0.2253, 0.1116, 0.1045],
        # [0.2341, 0.1148, 0.0975],
        # [0.2219, 0.1030, 0.0840]]
        return self.final_softmax(x)

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

        def combined_loss(output, target):
            '''
            Args:
                output:
                    {
                        "seg": predicted segs
                        "classification": predicted classes. should be torch.tensor(
                            [a,b,c],
                            ...
                            [a,b,c]
                        )
                    }
                target:
                    {
                        "seg": actual segmentation
                        "classification": actual class (integer)
                    }
            '''
            seg_output = output["seg"]
            cls_output = output["classification"]

            target_seg = target['seg']
            target_cls = target['classification']
            # for each c
            target_cls = torch.tensor(
                [
                    [1.0 if t == x else 0.0 for t in range(0,3)]
                    for x in target['classification']
                ]
            ) 
            # print(target_cls)

            seg_loss_value = seg_loss(seg_output, target_seg)
            cls_loss_value = self.f1_loss(cls_output, target_cls)
            total_loss = seg_loss_value + 0.5 * cls_loss_value  # Weighted combination

            return total_loss, seg_loss_value, cls_loss_value

        return combined_loss

    def f1_loss(self, y_pred, y_true):
        """
        Compute F1 loss for classification.
        """
        y_pred = torch.argmax(y_pred, dim=1)
        tp = (y_pred * y_true).sum().to(torch.float32)
        tn = ((1 - y_pred) * (1 - y_true)).sum().to(torch.float32)
        fp = (y_pred * (1 - y_true)).sum().to(torch.float32)
        fn = ((1 - y_pred) * y_true).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return 1 - f1

    def train_step(self, data_dict):
        """
        Perform a training step with multi-task loss.
        """
        input = data_dict['data']  # Input images
        expected_segs = data_dict['target']
        expected_classifications = [int(x.split('_')[1]) for x in data_dict['keys']] # gets the classifictions
        target = {
            'seg': expected_segs,  # Segmentation labels
            'classification': expected_classifications
        }
        # print(input.shape)
        # print("INPUT SHAPE")
        # INPUT SHAPE is usually torch.Size([3, 1, 64, 128, 192]) (3 is batch size)

        # Forward pass
        output = self.network.encoder(input) # get encoder output
        # unsure why output isn't already a tensor..?
        # print(output)
       
        # print(output[-1].shape)
        # print("ENCODER OUTPUT SHAPE")
        segmentation_pred = self.network.decoder(output)

        # print(segmentation_pred[-1].shape)
        # print(segmentation_pred)
        # print("DECODER OUTPUT")
        # classification. choose the last from the output as that is the final output from the encoder
        classification_pred = self.classification_head(output[-1])
        # print("CLASSIFICATION OUTPUTS:")
        # print(classification_pred)
        total_output = {
            "seg": segmentation_pred,
            "classification": classification_pred
        }

        total_loss, seg_loss, cls_loss = self.loss(total_output, target)

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
        input = data_dict['data']  # Input images
        expected_segs = data_dict['target']
        expected_classifications = [int(x.split('_')[1]) for x in data_dict['keys']] # gets the classifictions
        target = {
            'seg': expected_segs,  # Segmentation labels
            'classification': expected_classifications
        }

        # Forward pass
        output = self.network(input)
        total_loss, seg_loss, cls_loss = self.loss(output, target)

        return {
            "total_loss": total_loss.item(),
            "seg_loss": seg_loss.item(),
            "cls_loss": cls_loss.item(),
        }
