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
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(
        self, 
        input_channels, 
        num_classes=3, 
        conv_configs=None,
        dropout_rate=0.3
    ):
        super(ClassificationHead, self).__init__()
        
        # Default conv configurations if not provided
        if conv_configs is None:
            conv_configs = [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}
            ]
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        
        for config in conv_configs:
            conv_layer = nn.Sequential(
                nn.Conv3d(
                    current_channels, 
                    config['out_channels'], 
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', 1),
                    padding=config.get('padding', 0)
                ),
                nn.BatchNorm3d(config['out_channels']),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout_rate)
            )
            self.conv_layers.append(conv_layer)
            current_channels = config['out_channels']
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(current_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
      
    def forward(self, x):
        # x = x.to(self.device)
        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)
        
        # Classification
        return F.softmax(self.fc_layers(x), dim=1)

class nnUNetMultiTaskTrainer(nnUNetTrainer):
    """
    Custom trainer for multi-task learning: segmentation + classification.
    """
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans,
configuration, fold, dataset_json, unpack_dataset, device)
        self.num_classes_classification = 3  # Set the number of classes for classification
        self.classification_head = None
        self.classification_optimizer = None
        self.focal_gamma = 2
        self.use_focal_loss = True

        self.save_every = 5

    def initialize(self):
        """
        Extend the base network with a classification head.
        """
        super().initialize()  # Initialize the base network
        encoder_output_channels = self.network.encoder.output_channels[-1]
        self.classification_head = ClassificationHead(encoder_output_channels, self.num_classes_classification).to(self.device)

        self.classification_optimizer = torch.optim.Adam(self.classification_head.parameters(), lr=0.01)
        self.save_every = 5
        print("SAVING MODEL EVERY N EPOCHS BELOW:")
        print(self.save_every)


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
            target_seg = target['seg']

            seg_loss_weight = 0.8
            cls_loss_weight = 0.2
            
            # Soft Dice Loss
            seg_loss = self._compute_soft_dice_loss(seg_output, target_seg)
            
            # Classification Loss
            cls_output = output["classification"]
            target_cls = torch.tensor(target['classification'], 
                                      device=cls_output.device, 
                                      dtype=torch.long)
            
            # Focal Cross Entropy Loss
            # if self.use_focal_loss:
            cls_loss = self._focal_cross_entropy(cls_output, target_cls)
            # else:
                # cls_loss = F.cross_entropy(cls_output, target_cls)
            
            # Weighted combination of losses
            total_loss = (seg_loss_weight * seg_loss + 
                          cls_loss_weight * cls_loss)
            
            return total_loss, seg_loss, cls_loss

        return combined_loss

    def _compute_soft_dice_loss(self, pred_segs, target_segs, smooth=1e-5):
        """
        Compute soft Dice loss for deep supervision
        
        Args:
            pred_segs (list): Predicted segmentation maps at different scales
            target_segs (list): Ground truth segmentation maps
            smooth (float): Smoothing factor to prevent division by zero
        
        Returns:
            torch.Tensor: Soft Dice loss
        """
        dice_losses = []
        for pred, target in zip(pred_segs, target_segs):
            # Ensure prediction and target are in the same shape
            pred = pred.softmax(dim=1)
            target = target.long()
            target = target.squeeze(1)
            # print(len(target))
            # print(target.shape)
            target_one_hot = F.one_hot(target, num_classes=3).permute(0, 4, 1, 2, 3).float()
            
            intersection = torch.sum(pred * target_one_hot)
            union = torch.sum(pred) + torch.sum(target_one_hot)
            
            dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
            dice_losses.append(dice_loss)
        
        return torch.mean(torch.stack(dice_losses))
    
    def _focal_cross_entropy(self, input, target):
        """
        Focal cross-entropy loss to handle class imbalance
        
        Args:
            input (torch.Tensor): Raw model predictions
            target (torch.Tensor): Ground truth labels
        
        Returns:
            torch.Tensor: Focal cross-entropy loss
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return focal_loss

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
        
  

        classification_input = output[-1]
        classification_pred = self.classification_head(classification_input)
        # print("CLASSIFICATION OUTPUTS:")
        # print(classification_pred)
        total_output = {
            "seg": [pred.to(self.device) for pred in segmentation_pred],
            "classification": classification_pred
        }

        total_loss, seg_loss, cls_loss = self.loss(total_output, target)

        # Backpropagation and optimization (for encoder / decoder)
        # with torch.autograd.set_detect_anomaly(True):
          # print("BACKPROP FOR SEG")
        self.optimizer.zero_grad()
        seg_loss.backward(retain_graph=True)
        self.optimizer.step()

        # with torch.autograd.set_detect_anomaly(True):
          # print("BACKPROP FOR CLS")
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
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            print("DDP")
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            seg_loss = np.vstack(losses_tr).mean()
            cls_loss = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            seg_loss = np.mean(outputs['seg_loss'])
            cls_loss = np.mean(outputs['cls_loss'])

            print("TRAIN SEG LOSS: ")
            print(seg_loss)
            print("TRAIN CLS LOSS: ")
            print(cls_loss)

        self.logger.log('train_losses', loss_here, self.current_epoch)

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

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard, 'seg_loss': seg_loss.detach().cpu().numpy(), 'cls_loss': class_Loss.detach().cpu().numpy()}
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            print("DDP")
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

        else:
            loss_here = np.mean(outputs_collated['loss'])
            seg_loss = np.mean(outputs_collated['seg_loss'])
            cls_loss = np.mean(outputs_collated['cls_loss'])
            print("VAL SEG LOSS: ")
            print(seg_loss)
            print("VAL CLS LOSS: ")
            print(cls_loss)


        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

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

            if key.startswith('_orig_mod'):
                key = key[7:]
                print(key)
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

        self.was_initialized = True