import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

from ops.utils import count_conv2d_flops
from ops.utils import conv2d_out_dim
from ops.mask import Mask_s

# TODO YD: Extend dynamic Spatial-Temporal-Channel to larger models
#__all__ = ['BateNet', 'batenet18', 'batenet34', 'batenet50', 'batenet101', 'batenet152']
__all__ = ['ResDynSTCNet', 'res18_dynstc_net']

model_urls = {
    'res18_dynstc_net': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'batenet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'batenet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'batenet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'batenet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def list_sum(obj):
    if isinstance(obj, list):
        if len(obj)==0:
            return 0
        else:
            return sum(list_sum(x) for x in obj)
    else:
        return obj

def shift(x, n_segment, fold_div=3, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    if inplace:
        # Due to some out of order error when performing parallel computing.
        # May need to write a CUDA kernel.
        raise NotImplementedError
        # out = InplaceShift.apply(x, fold)
    else:
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)

class PolicyBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer, shared, args):
        super(PolicyBlock, self).__init__()
        self.args = args
        self.norm_layer = norm_layer
        self.shared = shared
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        self.action_dim = out_factor
        self.flops = 0.0 # flops calculation supported only for 2-layer fully connected network

        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim * (args.gate_channel_ends - args.gate_channel_starts) // 64
        self.num_channels = out_dim // self.action_dim

        keyword = "%d_%d" % (in_dim, out_dim)
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim

        if self.args.gate_conv_embed_type != "None":
            if self.args.gate_conv_embed_type == "conv3x3":
                self.gate_conv_embed = conv3x3(in_planes, in_planes, stride=1, groups=1, dilation=1)
            elif self.args.gate_conv_embed_type == "conv3x3dw":
                self.gate_conv_embed = conv3x3(in_planes, in_planes, stride=1, groups=in_planes, dilation=1)
            elif self.args.gate_conv_embed_type == "conv1x1":
                self.gate_conv_embed = conv1x1(in_planes, in_planes, stride=1, groups=1, dilation=1)
            elif self.args.gate_conv_embed_type == "conv1x1dw":
                self.gate_conv_embed = conv1x1(in_planes, in_planes, stride=1, groups=in_planes, dilation=1)
            self.gate_conv_embed_bn = nn.BatchNorm2d(in_planes)
            self.gate_conv_embed_relu = nn.ReLU(inplace=True)

        if self.args.single_linear:
            self.gate_fc0 = nn.Linear(in_dim, out_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(out_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
        elif self.args.triple_linear:
            self.gate_fc0 = nn.Linear(in_planes * in_factor, hidden_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim)

            if self.args.gate_bn_between_fcs:
                self.gate_bn1 = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu1 = nn.ReLU(inplace=True)
            self.gate_fc2 = nn.Linear(hidden_dim, out_dim)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)
            normal_(self.gate_fc2.weight, 0, 0.001)
            constant_(self.gate_fc2.bias, 0)

        else:
            if self.args.shared_policy_net:
                self.gate_fc0 = self.shared[0][keyword]
            else:
                self.gate_fc0 = nn.Linear(in_dim, hidden_dim)
                self.flops = self.get_flops(in_dim, hidden_dim, out_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)
            if self.args.shared_policy_net:
                self.gate_fc1 = self.shared[1][keyword]
            else:
                self.gate_fc1 = nn.Linear(hidden_dim, out_dim)

            if not self.args.shared_policy_net:
                normal_(self.gate_fc0.weight, 0, 0.001)
                constant_(self.gate_fc0.bias, 0)
                normal_(self.gate_fc1.weight, 0, 0.001)
                constant_(self.gate_fc1.bias, 0)

    def get_flops(self, in_dim, hidden_dim, out_dim):
        return hidden_dim * in_dim + out_dim * hidden_dim

    def forward(self, x, **kwargs):
        # data preparation
        if self.args.gate_conv_embed_type != "None":
            x_input = self.gate_conv_embed(x)
            x_input = self.gate_conv_embed_bn(x_input)
            x_input = self.gate_conv_embed_relu(x_input)
        else:
            x_input = x

        if self.args.gate_reduce_type=="avg":
            x_c = nn.AdaptiveAvgPool2d((1, 1))(x_input)
        elif self.args.gate_reduce_type=="max":
            x_c = nn.AdaptiveMaxPool2d((1, 1))(x_input)
        x_c = torch.flatten(x_c, 1)
        _nt, _c = x_c.shape
        _t = self.args.num_segments
        _n = _nt // _t

        # history
        if self.args.gate_history:
            x_c_reshape = x_c.view(_n, _t, _c)
            h_vec = torch.zeros_like(x_c_reshape)
            h_vec[:, 1:] = x_c_reshape[:, :-1]
            h_vec = h_vec.view(_nt, _c)
            x_c = torch.cat([h_vec, x_c], dim=-1)

        # fully-connected embedding
        if self.args.single_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)

        elif self.args.triple_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn1(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu1(x_c)
            x_c = self.gate_fc2(x_c)

        else:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

        # gating operations
        x_c2d = x_c.view(x.shape[0], self.num_channels // self.args.granularity, self.action_dim)
        x_c2d = torch.log(F.softmax(x_c2d, dim=2).clamp(min=1e-8))
        mask = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=not self.args.gate_gumbel_use_soft)

        if self.args.granularity>1:
            mask = mask.repeat(1, self.args.granularity, 1)
        if self.args.gate_channel_starts>0 or self.args.gate_channel_ends<64:
            full_channels = mask.shape[1] // (self.args.gate_channel_ends-self.args.gate_channel_starts) * 64
            channel_starts = full_channels // 64 * self.args.gate_channel_starts
            channel_ends = full_channels // 64 * self.args.gate_channel_ends
            outer_mask = torch.zeros(mask.shape[0], full_channels, mask.shape[2]).to(mask.device)
            outer_mask[:, :, -1] = 1.
            outer_mask[:, channel_starts:channel_ends] = mask

            return outer_mask
        else:
            return mask  # TODO: BT*C*ACT_DIM


def handcraft_policy_for_masks(x, out, num_channels, use_current, args):
    factor = 3 if args.gate_history else 2

    if use_current:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.

    elif args.gate_all_zero_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_all_one_policy:
        mask = torch.ones(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_only_current_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.
    elif args.gate_random_soft_policy:
        mask = torch.rand(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_random_hard_policy:
        tmp_value = torch.rand(x.shape[0], num_channels, device=x.device)
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)

        if len(args.gate_stoc_ratio) > 0:
            _ratio = args.gate_stoc_ratio
        else:
            _ratio = [0.333, 0.333, 0.334] if args.gate_history else [0.5, 0.5]
        mask[:, :, 0][torch.where(tmp_value < _ratio[0])] = 1
        if args.gate_history:
            mask[:, :, 1][torch.where((tmp_value < _ratio[1] + _ratio[0]) & (tmp_value > _ratio[0]))] = 1
            mask[:, :, 2][torch.where(tmp_value > _ratio[1] + _ratio[0])] = 1

    elif args.gate_threshold:
        stat = torch.norm(out, dim=[2, 3], p=1) / out.shape[2] / out.shape[3]
        mask = torch.ones_like(stat).float()
        if args.absolute_threshold is not None:
            mask[torch.where(stat < args.absolute_threshold)] = 0
        else:
            if args.relative_max_threshold is not None:
                mask[torch.where(
                    stat < torch.max(stat, dim=1)[0].unsqueeze(-1) * args.relative_max_threshold)] = 0
            else:
                mask = torch.zeros_like(stat)
                c_ids = torch.topk(stat, k=int(mask.shape[1] * args.relative_keep_threshold), dim=1)[1]  # TODO B*K
                b_ids = torch.tensor([iii for iii in range(mask.shape[0])]).to(mask.device).unsqueeze(-1).expand(c_ids.shape)  # TODO B*K
                mask[b_ids.detach().flatten(), c_ids.detach().flatten()] = 1

        mask = torch.stack([1 - mask, mask], dim=-1)

    return mask


def get_hmap(out, args, **kwargs):
    out_reshaped = out.view((-1, args.num_segments) + out.shape[1:])

    if args.gate_history:
        h_map_reshaped = torch.zeros_like(out_reshaped)
        h_map_reshaped[:, 1:] = out_reshaped[:, :-1]
    else:
        return None

    if args.gate_history_detach:
        h_map_reshaped = h_map_reshaped.detach()

    h_map_updated = h_map_reshaped.view((-1,) + out_reshaped.shape[2:])
    return h_map_updated


def fuse_out_with_mask(out, mask, mask_s, h_map, apply_mask_s, args):
    if mask is not None:
        out = out * mask[:, :, -1].unsqueeze(-1).unsqueeze(-1)
        if args.gate_history:
            out = out + h_map * mask[:, :, -2].unsqueeze(-1).unsqueeze(-1)
    if apply_mask_s:
        out = out * mask_s

    return out


def count_dyn_conv2d_flops(input_data_shape, conv, spatial_mask, channels_mask, upstream_conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0

    # compute precise GFLOP
    out_active_pixels = torch.sum(spatial_mask, dim=(2, 3, 4))  # [batch_size, frames_per_clip]
    frames_per_clip = channels_mask.shape[1]
    gate_history = channels_mask.shape[-1] > 2
    # flops_per_frame: [batch_size, frames_per_clip]
    # flops_upper_bound_mat: [batch_size, frames_per_clip]
    if upstream_conv:
        # batch_size*frames*channels*K->batch_size*frames*channels
        out_channel_off = torch.zeros_like(channels_mask[:, :, :, 0], device=spatial_mask.device)
        # TODO YD: vectorize
        for t in range(frames_per_clip - 1):
            if gate_history:
                out_channel_off[:, t, :] = (1 - channels_mask[:, t, :, -1]) * (1 - channels_mask[:, t + 1, :, -2])
            else:
                out_channel_off[:, t, :] = 1 - channels_mask[:, t, :, -1]  # since no reusing, as long as not keeping, save from upstream conv
        out_channel_off[:, -1, :] = 1 - channels_mask[:, t, :, -1]  # TODO YD: this should probably be t+1
        out_active_channels = c_out - torch.sum(out_channel_off, dim=2)  # [batch_size, frames_per_clip]
        # TODO YD: verify the below for bottleneck
        flops_per_frame = out_active_channels * out_active_pixels * \
                          (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
        flops_upper_bound_mat = c_out * out_active_pixels * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    else:
        # downstream conv flops saving is from skippings
        in_channel_off = channels_mask[:, :, :, 0]  # [batch_size, frames_per_clip, channels]
        in_active_channels = torch.tensor([c_in], device=channels_mask.device, dtype=torch.float) - \
                             torch.sum(in_channel_off, dim=2)  # [batch_size, frames_per_clip]
        # TODO YD: verify the below for bottleneck
        flops_per_frame = c_out * out_active_pixels * \
                          (in_active_channels // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
        flops_upper_bound_mat = flops_per_frame

    flops = flops_per_frame.reshape((-1,))
    flops_upper_bound = torch.sum(flops_upper_bound_mat, dim=1) # [batch_size,]

    return flops, flops_upper_bound, (n, c_out, h_out, w_out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, h, w, mask_tile=8, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # spatial gating module
        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width  = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        # TODO YD: Check regarding bias=0 (not in original code but bias appears in trained model)
        self.mask_s = Mask_s(self.height, self.width, inplanes, mask_tile, mask_tile, bias=0, args=args)
        self.upsample = nn.Upsample(size=(self.height, self.width), mode='nearest')
        self.mask_tile = mask_tile
        # conv 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv 2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # misc
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        self.shall_enable = shall_enable
        self.num_channels = planes
        self.adaptive_policy = not any([self.args.gate_all_zero_policy,
                                        self.args.gate_all_one_policy,
                                        self.args.gate_only_current_policy,
                                        self.args.gate_random_soft_policy,
                                        self.args.gate_random_hard_policy,
                                        self.args.gate_threshold])
        if self.shall_enable==False and self.adaptive_policy:
            self.adaptive_policy=False
            self.use_current=True
        else:
            self.use_current=False

        if self.adaptive_policy:
            self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=planes, norm_layer=norm_layer, shared=shared, args=args)

            if self.args.dense_in_block:
                self.policy_net2 = PolicyBlock(in_planes=planes, out_planes=planes, norm_layer=norm_layer, shared=shared, args=args)

        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            self.gate_hist_conv = conv3x3(planes, planes, groups=planes)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                self.gate_hist_bnrelu = nn.Sequential(norm_layer(planes), nn.ReLU(inplace=True))
        # flops
        input_data_shape = (args.num_segments, inplanes, h, w)
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        flops_conv1_full = torch.Tensor([conv1_flops])
        flops_conv2_full = torch.Tensor([conv2_flops])
        self.flops_downsample = torch.Tensor([downsample0_flops])
        self.flops_channels_policy = torch.Tensor([self.policy_net.flops * args.num_segments])
        self.flops_full = flops_conv1_full + flops_conv2_full + self.flops_downsample + self.flops_channels_policy
        # mask flops
        flops_mks = self.mask_s.get_flops()
        #flops_mkc = self.mask_c.get_flops()
        self.flops_mask = torch.Tensor([flops_mks])

    def count_flops(self, input_data_shape, **kwargs):
        # TODO YD: This is kept as a ref for comparison with new code
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0
        return [conv1_flops, conv2_flops, downsample0_flops, 0], conv2_out_shape

    def forward(self, input, **kwargs):
        x, norm_spatial, flops = input
        identity = x

        # shift operations
        if self.args.shift:
            x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)

        mask_s = torch.ones(x.shape[0], 1, self.height, self.width, device=x.device)
        norm_s = torch.ones(x.shape[0], dtype=torch.float, device=x.device) * (self.height // self.mask_tile) * (self.width // self.mask_tile)
        norm_s_t = torch.tensor([(self.height // self.mask_tile) * (self.width // self.mask_tile)], dtype=torch.float, device=x.device)
        if self.args.spatial_masking:
            mask_s_m, norm_s, norm_s_t = self.mask_s(x) # [N, 1, h, w]
            mask_s = self.upsample(mask_s_m) # [N, 1, H, W]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # gate functions
        h_map_updated = get_hmap(out, self.args, **kwargs)
        if self.adaptive_policy:
            mask = self.policy_net(x, **kwargs)
        else:
            mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)

        apply_spatial_mask_in_conv1 = self.args.spatial_masking and not self.training
        out = fuse_out_with_mask(out, mask, mask_s, h_map_updated, apply_spatial_mask_in_conv1, self.args)

        x2 = out
        out = self.conv2(out)
        out = self.bn2(out)

        # gate functions
        if self.args.dense_in_block:
            h_map_updated2 = get_hmap(out, self.args, **kwargs)
            if self.adaptive_policy:
                mask2 = self.policy_net2(x2, **kwargs)
            else:
                mask2 = handcraft_policy_for_masks(x2, out, self.num_channels, self.use_current, self.args)
            # TODO YD: check removal of "dense_in_block" option
            out = fuse_out_with_mask(out, mask2, mask_s, h_map_updated2, self.args.spatial_masking, self.args)
            mask2 = mask2.view((-1, self.args.num_segments) + mask2.shape[1:])
        else:
            out = fuse_out_with_mask(out, mask=None, mask_s=mask_s, h_map=None, apply_mask_s=self.args.spatial_masking,
                                     args=self.args)
            mask2 = None

        if self.downsample0 is not None:
            y = self.downsample0(x)
            identity = self.downsample1(y)
        out += identity
        out = self.relu(out)
        # norm
        norm_spatial = torch.cat((norm_spatial, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
        # flops
        flops_blk = self.get_flops(mask_s, mask, x.shape)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))

        return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2,\
               mask_s.view((-1, self.args.num_segments, 1)  + mask_s.shape[2:]), norm_spatial, flops

    def get_flops(self, mask_s_up, mask_c, input_data_shape):
        # mask_s_up: [NumFrames, 1, conv1_out_height, conv1_out_width]
        # mask_c: [NumFrames, NumChannels, 3] -> 3: col0:skip,col1:reuse,col2:keep
        spatial_masks = mask_s_up.view((-1, self.args.num_segments) + mask_s_up.shape[1:]) #[BatchesNum, FramesPerClip, 1, conv1_out_height, conv1_out_width]
        channels_masks = mask_c.view((-1, self.args.num_segments) + mask_c.shape[1:]) #[BatchesNum, FramesPerClip, NumChannels, 3]
        # conv1
        flops_conv1, flops_upper_bound_conv1, conv1_out_shape = count_dyn_conv2d_flops(input_data_shape, self.conv1,
                                                                                       spatial_masks, channels_masks,
                                                                                       upstream_conv=True)
        # conv2
        flops_conv2, flops_upper_bound_conv2, conv2_out_shape = count_dyn_conv2d_flops(conv1_out_shape, self.conv2,
                                                                                       spatial_masks, channels_masks,
                                                                                       upstream_conv=False)
        # total
        flops = flops_conv1 + flops_conv2
        flops_upper_bound = torch.unsqueeze(torch.sum(flops_upper_bound_conv1 + flops_upper_bound_conv2), 0)
        # flops: [1, batch_size x frames_per_clip]
        # flops upper bound: scalar
        return torch.cat((flops, flops_upper_bound, self.flops_downsample.to(flops.device),
                          self.flops_channels_policy.to(flops.device), self.flops_mask.to(flops.device),
                          self.flops_full.to(flops.device)))


# TODO YD: Extend dynamic Spatial-Temporal-Channel to larger models
# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample0 = downsample0
#         self.downsample1 = downsample1
#         self.stride = stride
#
#         self.args = args
#         self.shall_enable = shall_enable
#         self.num_channels = width
#         self.adaptive_policy = not any([self.args.gate_all_zero_policy,
#                                         self.args.gate_all_one_policy,
#                                         self.args.gate_only_current_policy,
#                                         self.args.gate_random_soft_policy,
#                                         self.args.gate_random_hard_policy,
#                                         self.args.gate_threshold])
#         if self.shall_enable==False and self.adaptive_policy:
#             self.adaptive_policy=False
#             self.use_current=True
#         else:
#             self.use_current=False
#
#         if self.adaptive_policy:
#             self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=width, norm_layer=norm_layer, shared=shared, args=args)
#             if self.args.dense_in_block:
#                 self.policy_net2 = PolicyBlock(in_planes=width, out_planes=width, norm_layer=norm_layer, shared=shared, args=args)
#
#         if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
#             self.gate_hist_conv = conv3x3(width, width, groups=width)
#             if self.args.gate_history_conv_type == 'ghostbnrelu':
#                 self.gate_hist_bnrelu = nn.Sequential(norm_layer(width), nn.ReLU(inplace=True))
#
#     def count_flops(self, input_data_shape, **kwargs):
#         conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
#         conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
#         conv3_flops, conv3_out_shape = count_conv2d_flops(conv2_out_shape, self.conv3)
#         if self.downsample0 is not None:
#             downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
#         else:
#             downsample0_flops = 0
#
#         return [conv1_flops, conv2_flops, conv3_flops, downsample0_flops, 0], conv3_out_shape
#
#     def forward(self, x, **kwargs):
#         identity = x
#
#         # shift operations
#         if self.args.shift:
#             x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         # gate functions
#         h_map_updated = get_hmap(out, self.args, **kwargs)
#         if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
#             h_map_updated = self.gate_hist_conv(h_map_updated)
#             if self.args.gate_history_conv_type == 'ghostbnrelu':
#                 h_map_updated = self.gate_hist_bnrelu(h_map_updated)
#
#         if self.adaptive_policy:
#             mask = self.policy_net(x, **kwargs)
#         else:
#             mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)
#
#         out = fuse_out_with_mask(out, mask, h_map_updated, self.args)
#
#         x2 = out
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         # gate functions
#         if self.args.dense_in_block:
#             h_map_updated2 = get_hmap(out, self.args, **kwargs)
#             if self.adaptive_policy:
#                 mask2 = self.policy_net2(x2, **kwargs)
#             else:
#                 mask2 = handcraft_policy_for_masks(x2, out, self.num_channels, self.use_current, self.args)
#             out = fuse_out_with_mask(out, mask2, h_map_updated2, self.args)
#             mask2 = mask2.view((-1, self.args.num_segments) + mask2.shape[1:])
#         else:
#             mask2 = None
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample0 is not None:
#             y = self.downsample0(x)
#             identity = self.downsample1(y)
#
#         out += identity
#         out = self.relu(out)
#
#         return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2


class ResDynSTCNet(nn.Module):

    def __init__(self, block, layers, h=224, w=224, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, args=None):
        super(ResDynSTCNet, self).__init__()
        # block
        self.height, self.width = h, w
        # norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # conv1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.args = args

        if self.args.shared_policy_net:
            self.gate_fc0s = nn.ModuleDict()
            self.gate_fc1s = nn.ModuleDict()
        else:
            self.gate_fc0s = None
            self.gate_fc1s = None

        self.relu = nn.ReLU(inplace=True)
        h = conv2d_out_dim(h, kernel_size=7, stride=2, padding=3)
        w = conv2d_out_dim(w, kernel_size=7, stride=2, padding=3)
        self.flops_conv1 = torch.Tensor([49 * h * w * self.inplanes * 3])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)
        # residual blocks
        self.layer1, h, w = self._make_layer(block, 64 * 1, layers[0], h, w, 8, layer_offset=0)
        self.layer2, h, w = self._make_layer(block, 64 * 2, layers[1], h, w, 4,
                                       stride=2, dilate=replace_stride_with_dilation[0], layer_offset=layers[0])
        self.layer3, h, w = self._make_layer(block, 64 * 4, layers[2], h, w, 2,
                                       stride=2, dilate=replace_stride_with_dilation[1], layer_offset=layers[0] + layers[1])
        self.layer4, h, w = self._make_layer(block, 64 * 8, layers[3], h, w, 1,
                                       stride=2, dilate=replace_stride_with_dilation[2], layer_offset=layers[0] + layers[1] + layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.args.policy_attention:
            self.attention_fc0 = nn.Linear(16*3, 16)
            self.attention_relu = nn.ReLU(inplace=True)
            self.attention_bn = nn.BatchNorm1d(16)
            self.attention_fc1 = nn.Linear(16, 1)
            normal_(self.attention_fc0.weight, 0, 0.001)
            constant_(self.attention_fc0.bias, 0)
            normal_(self.attention_fc1.weight, 0, 0.001)
            constant_(self.attention_fc1.bias, 0)
            nn.init.constant_(self.attention_bn.weight, 1)
            nn.init.constant_(self.attention_bn.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        zero_init_residual = args.zero_init_residual
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # TODO YD: Extend dynamic Spatial-Temporal-Channel to larger models
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     # for bn in m.bn2s:
                #     #     nn.init.constant_(bn.weight, 0)
                #     nn.init.constant_(m.bn2.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def update_shared_net(self, in_planes, out_planes):
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim
        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim // 64 * (self.args.gate_channel_ends - self.args.gate_channel_starts)
        keyword = "%d_%d" % (in_dim, out_dim)
        if keyword not in self.gate_fc0s:
            self.gate_fc0s[keyword] = nn.Linear(in_dim, hidden_dim)
            self.gate_fc1s[keyword] = nn.Linear(hidden_dim, out_dim)
            normal_(self.gate_fc0s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc0s[keyword].bias, 0)
            normal_(self.gate_fc1s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc1s[keyword].bias, 0)

    def _make_layer(self, block, planes_list_0, blocks, h, w, mask_tile, stride=1, dilate=False, layer_offset=-1):
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1 = None
        previous_dilation = self.dilation
        # Spatial mask
        mask_s = torch.ones(blocks)
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list_0 * block.expansion:
            downsample0 = conv1x1(self.inplanes, planes_list_0 * block.expansion, stride)
            downsample1 = norm_layer(planes_list_0 * block.expansion)

        _d={1:0, 2:1, 4:2, 8:3}
        layer_idx = _d[planes_list_0//64]

        if len(self.args.enabled_layers) > 0:
            enable_policy = layer_offset in self.args.enabled_layers
            print("stage-%d layer-%d (abs: %d) enabled:%s"%(layer_idx, 0, layer_offset, enable_policy))
        elif len(self.args.enabled_stages) > 0:
            enable_policy = layer_idx in self.args.enabled_stages
        else:
            enable_policy = (layer_idx >= self.args.enable_from and layer_idx < self.args.disable_from)

        if self.args.shared_policy_net and enable_policy:
            self.update_shared_net(self.inplanes, planes_list_0)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list_0, h, w, mask_tile, stride, downsample0, downsample1, self.groups,
                            self.base_width, previous_dilation, norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        h = conv2d_out_dim(h, kernel_size=1, stride=stride, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=stride, padding=0)
        self.inplanes = planes_list_0 * block.expansion
        for k in range(1, blocks):

            if len(self.args.enabled_layers) > 0:
                enable_policy = layer_offset + k in self.args.enabled_layers
                print("stage-%d layer-%d (abs: %d) enabled:%s" % (layer_idx, k, layer_offset + k, enable_policy))

            if self.args.shared_policy_net and enable_policy:
                self.update_shared_net(self.inplanes, planes_list_0)

            layers.append(block(self.inplanes, planes_list_0, h, w, mask_tile, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        return layers, h, w

    def count_flops(self, input_data_shape, **kwargs):
        flops_list = []
        _B, _T, _C, _H, _W = input_data_shape
        input2d_shape = _B*_T, _C, _H, _W

        flops_conv1, data_shape = count_conv2d_flops(input2d_shape, self.conv1)
        data_shape = data_shape[0], data_shape[1], data_shape[2]//2, data_shape[3]//2 #TODO pooling
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                flops, data_shape = block.count_flops(data_shape, **kwargs)
                flops_list.append(flops)
        return flops_list

    def forward(self, input_data, **kwargs):
        # TODO x.shape (nt, c, h, w)
        frames_num, _, _, _ = input_data.shape

        if "tau" not in kwargs:
            kwargs["tau"] = 1
            kwargs["inline_test"] = True

        mask_stack_list = []  # TODO list for t-dimension
        spatial_masks_list = []
        for _, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for _, block in enumerate(layers):
                mask_stack_list.append(None)
                if self.args.dense_in_block:
                    mask_stack_list.append(None)
                spatial_masks_list.append(None)

        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        norm_spatial = torch.zeros(1, frames_num+1).to(x.device)
        flops = torch.zeros(1, frames_num+5).to(x.device)
        #+5 for 1)upperbound flops 2)downsample flops 3)channels mask flops
        #       4)spatial mask flops 5)orig layer flops (conv1+conv2+downsample)

        idx = 0
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                x, mask, mask2, mask_s, norm_spatial, flops = block((x, norm_spatial, flops), **kwargs)
                mask_stack_list[idx] = mask
                spatial_masks_list[idx] = mask_s
                idx += 1
                if self.args.dense_in_block:
                    mask_stack_list[idx] = mask2
                    idx += 1


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        # norm and flops
        norm_s = norm_spatial[1:, 0:frames_num].permute(1, 0).contiguous()
        norm_s_t = norm_spatial[1:, -1].unsqueeze(0)
        # The last element is a placeholder for the fully connected flops at the "clip level"
        flops_real = [flops[1:, 0:frames_num].permute(1, 0).contiguous(),
                      self.flops_conv1.to(x.device), torch.Tensor([0.0]).to(x.device)]
        flops_upperbound = flops[1:, -5].unsqueeze(0)
        flops_downsample = flops[1:, -4].unsqueeze(0)
        flops_channels_mask = flops[1:, -3].unsqueeze(0)
        flops_mask = flops[1:, -2].unsqueeze(0)
        flops_ori  = flops[1:, -1].unsqueeze(0)
        # get outputs
        dyn_outputs = {}
        # outputs["closs"], outputs["rloss"], outputs["bloss"] = self.get_loss(
        #                     x, label, batch_num, den_target, lbda, gamma, p,
        #                     norm_s, norm_c, norm_s_t, norm_c_t,
        #                     flops_real, flops_mask, flops_ori)
        dyn_outputs["flops_real"] = flops_real
        dyn_outputs["flops_upperbound"] = flops_upperbound
        dyn_outputs["flops_downsample"] = flops_downsample
        dyn_outputs["flops_channels_mask"] = flops_channels_mask
        dyn_outputs["flops_mask"] = flops_mask
        dyn_outputs["flops_ori"] = flops_ori
        dyn_outputs["norm_s"] = norm_s
        dyn_outputs["norm_s_t"] = norm_s_t

        if self.args.policy_attention:
            mask_stat_list=[]
            for mask_stack in mask_stack_list:
                mask_stat = torch.stack([torch.mean(mask_stack[:, :, :, act_i], dim=-1) for act_i in range(3)], dim=-1)
                mask_stat_list.append(mask_stat)
            mask_stat_tensor = torch.stack(mask_stat_list, dim=-2)
            att_input = mask_stat_tensor.view(x.shape[0], len(mask_stack_list) * 3)
            att_mid = self.attention_fc0(att_input)
            att_mid = self.attention_bn(att_mid)
            att_mid = self.attention_fc1(att_mid)
            attention = torch.softmax(att_mid.view(-1, self.args.num_segments), dim=-1).clamp(min=1e-8)

            # TODO YD: check if need policy attention + spatial masking support
            return out, mask_stack_list, attention
        else:
            return out, mask_stack_list, spatial_masks_list, dyn_outputs



def _resnet_dynstc(arch, block, layers, pretrained, progress, **kwargs):
    model = ResDynSTCNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        # TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
            # TODO layer4.0.downsample.X.weight -> layer4.0.downsampleX.weight
            if "downsample.0" in k:
                old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
            elif "downsample.1" in k:
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def res18_dynstc_net(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_dynstc('res18_dynstc_net', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)

# TODO YD: Extend dynamic Spatial-Temporal-Channel to larger models
# def batenet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _batenet('batenet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                     **kwargs)
#
#
# def batenet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _batenet('batenet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                     **kwargs)
#
#
# def batenet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _batenet('batenet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                     **kwargs)
#
#
# def batenet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _batenet('batenet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                     **kwargs)
