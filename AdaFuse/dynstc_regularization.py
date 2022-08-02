import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class spar_loss(nn.Module):
    def __init__(self):
        super(spar_loss, self).__init__()

    def forward(self, flops_real, flops_mask, flops_ori, batch_size, den_target, lbda):
        # TODO YD: remove - Dual-Gating sparsity loss:
        # # total sparsity
        # flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        # # block flops
        # flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
        # flops_mask = flops_mask.mean(0).sum()
        # flops_ori = flops_ori.mean(0).sum() + flops_conv1.mean() + flops_fc.mean()
        # flops_real = flops_conv + flops_mask + flops_conv1.mean() + flops_fc.mean()
        # # loss
        # rloss = lbda * (flops_real / flops_ori - den_target)**2

        # total sparsity
        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        frames_per_clip = flops_tensor.shape[1]
        # block flops
        flops_conv = flops_tensor.mean(dim=(0)).sum() # average flops of all resnet blocks per clip
        flops_mask = flops_mask.sum() * frames_per_clip # total flops for spatial mask generation in clip
        flops_ori = flops_ori.sum() + flops_conv1.mean() * frames_per_clip + flops_fc.mean() # original model flops per clip
        flops_real = flops_conv + flops_mask + flops_conv1.mean() * frames_per_clip + flops_fc.mean() # measured flops per clip
        # loss
        rloss = lbda * (flops_real / flops_ori - den_target)**2
        return rloss


class blance_loss(nn.Module):
    def __init__(self):
        super(blance_loss, self).__init__()

    def forward(self, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size, 
                den_target, gamma, p):
        # TODO YD: remove - Dual-Gating bound regularization:
        # norm_s = mask_norm_s
        # norm_s_t = norm_s_t.mean(0)
        # norm_c = mask_norm_c
        # norm_c_t = norm_c_t.mean(0)
        # den_s = norm_s[0:batch_size,:].mean(0) / norm_s_t
        # den_c = norm_c[0:batch_size,:].mean(0) / norm_c_t
        # den_tar = math.sqrt(den_target)
        # bloss_s = get_bloss_basic(den_s, den_tar, batch_size, gamma, p)
        # bloss_c = get_bloss_basic(den_c, den_tar, batch_size, gamma, p)
        # bloss = bloss_s + bloss_c

        norm_s = mask_norm_s
        norm_s_t = norm_s_t.mean(0)
        den_s = norm_s[0:batch_size,:].mean(dim=(0,1)) / norm_s_t
        den_tar = math.sqrt(den_target)
        bloss_s = get_bloss_basic(den_s, den_tar, batch_size, gamma, p)
        bloss = bloss_s

        return bloss


def get_bloss_basic(spar, spar_tar, batch_size, gamma, p):
    # bound
    bloss_l = (F.relu(p*spar_tar-spar)**2).mean()
    bloss_u = (F.relu(spar-1+p-p*spar_tar)**2).mean()
    bloss = gamma * (bloss_l + bloss_u)
    return bloss 


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss()
        self.spar_loss = spar_loss()
        self.balance_loss = blance_loss()
    
    def forward(self, output, targets, flops_real, flops_mask, flops_ori, flops_downsample, batch_size,
                den_target, lbda, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t,
                gamma, p):
        # output: [batch_size, num_classes] - model prediction per clip
        # targets: [batch_size, ] - ground truth class label per clip
        # flops_real - list:
        # flops_real[0]: [BatchSize, FramesPerClip, #ResNetLayers] - The actual measured flops per frame per residual block
        # flops_real[1]: scalar - flops (per 1 frame) of the first convolution in the network
        # flops_real[2]: scalar - flops (per clip) of the FC layer
        # flops_mask : [1, #ResNetLayers] - flops (per 1 frame) of the spatial mask generating conv at each ResNetLayer
        # flops_ori: [1, #ResNetLayers] - flops (per clip) of the static model ops (conv1+conv2+downsampling+channels policy) at each ResNetLayer
        # flops_downsample: [1, #ResNetLayers] - flops (per clip) of the downsampling (if exists) at each ResNetLayer
        # batch_size: - scalar - number of clips in batch
        # mask_norm_s: - [BatchSize, FramesPerClip,  # ResNetLayers] - norm of the spatial mask at each frame and resnet layer
        # norm_s_t:  [1, #ResNetLayers] - number of elements in the generated spatial mask at each ResNetLayer
        closs = self.task_loss(output, targets)
        sloss = self.spar_loss(flops_real, flops_mask, flops_ori, batch_size, den_target, lbda)
        bloss = self.balance_loss(mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size,
                                  den_target, gamma, p)
        return closs, sloss, bloss
