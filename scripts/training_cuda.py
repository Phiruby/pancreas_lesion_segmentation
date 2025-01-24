%%writefile /usr/local/lib/python3.11/dist-packages/nnunetv2/training/nnUNetTrainer/nnUNetMultiTaskTrainer.py
import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import distributed as dist
from typing import Tuple, Union, List
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from torch._dynamo import OptimizedModule

class ClassificationHead(nn.Module):
    """
    A simple classification head for multi-task learning.
    """
    def __init__(self, input_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global average pooling for 3D data
        self.fc = nn.Linear(input_channels, num_classes)
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        # need to padd due to uneven sizing
        x = self.global_pool(x).view(x.size(0), -1)  # Flatten after pooling
        x = self.fc(x)
        return self.sm(x)

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

        self.classification_optimizer = torch.optim.Adam(self.classification_head.parameters(), lr=1e-4)

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
            # print(len(target_seg))
            # for i in target_seg:
            #   print(i.shape)
            # print(len(target_cls))
            # print(target_seg[0].shape)
            # print(target_seg[1].shape)
            # print("^^^^^")
            # for each c
            predicted_cls = torch.argmax(cls_output, dim = 1)
            seg_loss_value = seg_loss(seg_output, target_seg)
            cls_loss_value = self.f1_loss(cls_output.to(self.device), torch.tensor(target_cls).to(self.device))
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
            'seg': [x.to(self.device) for x in expected_segs],  # Segmentation labels. Move to device
            'classification': expected_classifications
        }
        # print("EXPECTED SEG SHAPES")
        # for i in expected_segs:
          # print(i.shape)
        # print(input.shape)
        # print("INPUT SHAPE")
        # input to cuda
        input = input.to(self.device)
        # target['seg'] = target['seg'].to(self.device)
        # target['classification'] = torch.tensor(target['classification']).to(self.device)
        # INPUT SHAPE is usually torch.Size([3, 1, 64, 128, 192]) (3 is batch size)

        # Forward pass
        output = self.network.encoder(input) # get encoder output
        # unsure why output isn't already a tensor..?
        # print(output)

        # print(output[-1].shape)
        # print("ENCODER OUTPUT SHAPE")
        segmentation_pred = self.network.decoder(output)
        # print("PREDICTED SEG SHAPES:")
        # for i in segmentation_pred:
          # print(i.shape)
      
        # classification. choose the last from the output as that is the final output from the encoder
        # detach so backprop does
        classification_pred = self.classification_head(output[-1].to(self.device).detach())
        # print("CLASSIFICATION OUTPUTS:")
        # print(classification_pred)
        total_output = {
            "seg": [pred.to(self.device) for pred in segmentation_pred],
            "classification": classification_pred
        }

        total_loss, seg_loss, cls_loss = self.loss(total_output, target)

      
        # Separate backward passes
        # Segmentation loss backward (with existing optimizer)
        self.optimizer.zero_grad()
        seg_loss.backward(retain_graph=True)
        self.optimizer.step()

        # Classification loss backward
        self.classification_optimizer.zero_grad()
        cls_loss.backward()
        self.classification_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "seg_loss": seg_loss.item(),
            "cls_loss": cls_loss.item(),
            "loss": total_loss.item()
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        expected_classifications = [int(x.split('_')[1]) for x in batch['keys']] # gets the classifictions
        total_target = {
            'seg': [x.to(self.device) for x in target],  # Segmentation labels. Move to device
            'classification': expected_classifications
        }

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            enc_out = self.network.encoder(data)
            output = self.network.decoder(enc_out)
            clas_out = self.classification_head(enc_out[-1].to(self.device))
            total_output = {
                "seg": [pred.to(self.device) for pred in output],
                "classification": clas_out
            }
            del data
            l, seg_loss, class_Loss = self.loss(total_output, total_target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def save_checkpoint(self, filename: str) -> None:
        print("SAVING CHECKPOINT")

        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    print("INCLUDING CLASSIFICATION HEAD")
                    mod = self.network
                    classifier = self.classification_head

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'classification_optimizer_state': self.classification_optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'classification_head_weights': classifier.state_dict() if classifier is not None else None
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        
        print("Checkpoint keys: ", checkpoint.keys())
        if 'clssification_head_weights' in checkpoint.keys():
            print("Loading classification head!")
            self.classification_head.load_state_dict(checkpoint['classification_head_weights'])

        if 'classification_optimizer_state' in checkpoint:
            self.classification_optimizer.load_state_dict(checkpoint['classification_optimizer_state'])

        self.was_initialized = True