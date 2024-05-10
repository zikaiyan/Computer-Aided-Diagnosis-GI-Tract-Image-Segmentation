import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import directed_hausdorff

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input_flat = input.reshape(input.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        
        intersection = (input_flat * target_flat).sum(1)
        union = input_flat.sum(1) + target_flat.sum(1)
        
        dice = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()

class ChannelwiseHausdorffDistanceLoss(nn.Module):
    def __init__(self):
        super(ChannelwiseHausdorffDistanceLoss, self).__init__()

    def forward(self, input, target):
        batch_size = input.size(0)
        channels = input.size(1)
        hausdorff_loss = 0.0
        
        for i in range(batch_size):
            channel_losses = []
            for ch in range(channels):
                # Reshape each channel of the input and target to a flat array
                input_np = input[i, ch, :, :].detach().cpu().numpy()
                target_np = target[i, ch, :, :].detach().cpu().numpy()
                
                # Calculate directed Hausdorff distance for this channel
                d1 = directed_hausdorff(input_np, target_np)[0]
                d2 = directed_hausdorff(target_np, input_np)[0]
                channel_losses.append(max(d1, d2))
            
            # Average or max the channel-wise Hausdorff distances
            # You can choose max, mean or any other aggregation depending on the use case
            hausdorff_loss += np.mean(channel_losses)
        
        return hausdorff_loss / batch_size

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.4, weight_hausdorff=0.6):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.hausdorff_loss = ChannelwiseHausdorffDistanceLoss()
        self.weight_dice = weight_dice
        self.weight_hausdorff = weight_hausdorff

    def forward(self, input, target):
        dice_loss = self.dice_loss(input, target)
        hausdorff_loss = self.hausdorff_loss(input, target)
        combined_loss = self.weight_dice * dice_loss + self.weight_hausdorff * hausdorff_loss
        return combined_loss

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, input, target):
        # Assumes input and target are of the same shape [batch_size, channels, height, width]
        batch_size = input.size(0)
        channels = input.size(1)
        iou_values_0 = []
        iou_values_1 = []
        iou_values_2 = []

        for i in range(batch_size):
            for ch in range(channels):
                input_flat = (input[i, ch] > 0.5).float()
                target_flat = (target[i, ch] > 0.5).float()
                
                if target_flat.sum() == 0:
                    continue # skip cases where the target is all zeros since we dont have ground truth

                if input_flat.sum() == 0 and target_flat.sum() == 0:
                    continue  # Skip channel if both input and target are all zeros

                intersection = (input_flat * target_flat).sum()
                union = (input_flat + target_flat).clamp(0, 1).sum()  # Use clamp to handle union calculation

                if union == 0:
                    iou = torch.tensor(0.)  # Avoid division by zero; can also return 1 if both masks are empty
                else:
                    iou = intersection / union

                # Append iou based on the channel
                if ch == 0:
                    iou_values_0.append(iou)
                elif ch == 1:
                    iou_values_1.append(iou)
                elif ch == 2:
                    iou_values_2.append(iou)

        # Calculate mean avoiding empty lists
        mean_iou_0 = torch.tensor(iou_values_0).mean() if iou_values_0 else torch.tensor(0.)
        mean_iou_1 = torch.tensor(iou_values_1).mean() if iou_values_1 else torch.tensor(0.)
        mean_iou_2 = torch.tensor(iou_values_2).mean() if iou_values_2 else torch.tensor(0.)

        return mean_iou_0, mean_iou_1, mean_iou_2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
