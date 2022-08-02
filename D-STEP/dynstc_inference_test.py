import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import multiprocessing
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, \
    init_gflops_table, compute_gflops_by_mask, adjust_learning_rate, ExpAnnealing
from opts import parser
from ops.my_logger import Logger
import numpy as np
import common
from os.path import join as ospj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from analysis import flops_calculations as flops_utils
from analysis import net_data_visualizations as vis_utils
from analysis import policy_nets_stats as policy_stats
from collections import OrderedDict
from dynstc_regularization import Loss

def inner_main(argv):
    # ---------------------------------------------------------
    # 1. Parse input arguments and initialize global data paths
    args = parser.parse_args()
    common.set_manual_data_path(args.data_path, args.exps_path)
    test_mode = (args.test_from != "")
    assert test_mode, "inference_analysis: expecting --test_from argument as input to specify model checkpoint"

    set_random_seed(args.random_seed, args)

    # ---------------------------------------------------------
    # 2. Load current dataset metadata and GT file paths
    args.num_class, args.train_list, args.val_list, args.root_path, prefix, args.train_folder_suffix,\
    args.val_folder_suffix = \
        dataset_config.return_dataset(args.dataset, args.data_path)

    # YD: load class names
    categories_names = []
    if args.categories_file_apth:
        with open(args.categories_file_apth) as f:
            lines = f.readlines()
        categories_names = [item.rstrip() for item in lines]
    # ---------------------------------------------------------
    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    logger = Logger()
    sys.stdout = logger

    # ---------------------------------------------------------
    # 3. Initialize the video recognition model
    model = TSN_Gate(args=args)

    # ---------------------------------------------------------
    # 4. Get FLOPs of the original (static) model
    base_model_gflops, gflops_list, g_meta = init_gflops_table(model, args)

    # YD: Pass to our functions for further analysis
    flops_utils.static_model_flops_analysis(model, base_model_gflops, gflops_list, args.static_flops_table_path)

    # ---------------------------------------------------------
    if args.gpus is None:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cpu()
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # ---------------------------------------------------------
    # 5. Load model parameters from checkpoint
    if "pth.tar" not in args.test_from:
        the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
    else:
        the_model_path = args.test_from
    the_model_path = common.EXPS_PATH + "/" + the_model_path
    if args.gpus is None:
        sd = torch.load(the_model_path, map_location=torch.device('cpu'))['state_dict']
    else:
        sd = torch.load(the_model_path)['state_dict']
    model_dict = model.state_dict()
    model_dict.update(sd)
    # YD: load DG-Net pretrained spatial policy networks
    if args.spatial_masking and args.dgnet_pretrained_path:
        DYNSTC_MODEL_PREFIX = 'module.base_model.'
        if args.gpus is None:
            dgnet_sd_all = torch.load(args.dgnet_pretrained_path, map_location=torch.device('cpu'))['state_dict']
        else:
            dgnet_sd_all = torch.load(args.dgnet_pretrained_path)['state_dict']
        # Remove all "non spatial mask" entries from dgnet pretrained model dict
        dgnet_sd_filt = dgnet_sd_all.copy()
        for dgnet_key in dgnet_sd_all.keys():
            if '.mask_s.' not in dgnet_key:
                del dgnet_sd_filt[dgnet_key]
        dgnet_sd = OrderedDict()
        for dgnet_key in dgnet_sd_filt.keys():
            if dgnet_key.endswith('.eta'):
                continue
            new_mask_s_key = DYNSTC_MODEL_PREFIX + dgnet_key
            assert new_mask_s_key in model_dict.keys(), 'expected %s field in model_dict before updating with' \
                                                        ' DGNet pretrained model' % new_mask_s_key
            dgnet_sd[new_mask_s_key] = dgnet_sd_filt[dgnet_key]
        model_dict.update(dgnet_sd)

    model.load_state_dict(model_dict)

    cudnn.benchmark = True

    val_loader = get_data_loaders(model, prefix, args)
    if args.gpus is None:
        criterion = torch.nn.CrossEntropyLoss().cpu()
    else:
         criterion = torch.nn.CrossEntropyLoss().cuda()

    # TODO YD: make the following loss the criterion for optimization
    my_criterion = Loss()

    exp_full_path = setup_log_directory(args.exp_header, test_mode, args, logger)

    map_record, mmap_record, prec_record, prec5_record = get_recorders(4)

    # YD:Prepare empty folders for analysis data
    if args.debug_frames_dir_path:
        if not os.path.exists(args.debug_frames_dir_path):
            os.makedirs(args.debug_frames_dir_path)
    # YD: Write model description to csv
    if args.model_desc_table_path:
        for model_tensor_name, model_tensor in model_dict.items():
            pass
            #print('%s: %s' % (model_tensor_name, np.array(model_tensor.shape)))

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, prec5, val_usage_str = \
                validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path,
                         categories_names, my_criterion, args)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)
            prec5_record.update(prec5)

            print('Best Prec@1: %.3f (epoch=%d) w. Prec@5: %.3f' % (
                prec_record.best_val, prec_record.best_at,
                prec5_record.at(prec_record.best_at)))

            # Evaluation - only run for one epoch
            break

    # after fininshing all the epochs
    if args.skip_log == False:
        os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                         "_mm_%.2f_a_%.2f_f.txt" % (mmap_record.best_val, prec_record.best_val)))



def build_dataflow(dataset, is_train, batch_size, workers, not_pin_memory):
    workers = min(workers, multiprocessing.cpu_count())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                              num_workers=workers, pin_memory=not not_pin_memory, sampler=None,
                                              drop_last=is_train)
    return data_loader


def get_data_loaders(model, prefix, args):

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(model.module.scale_size)),
        GroupCenterCrop(model.module.crop_size),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    val_dataset = TSNDataSet(args.root_path, args.val_list,
                             num_segments=args.num_segments,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=(val_transform, val_transform),
                             dense_sample=args.dense_sample,
                             dataset=args.dataset,
                             filelist_suffix=args.filelist_suffix,
                             folder_suffix=args.val_folder_suffix,
                             save_meta=args.save_meta)

    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.not_pin_memory)

    return val_loader


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path, class_names,
             my_criterion, args):
    batch_time, top1, top5 = get_average_meters(3)
    all_results = []
    all_targets = []

    tau = args.init_tau

    # TODO YD: update mask list to include spatial mask
    if "batenet" in args.arch or "AdaBNInc" in args.arch or "dynstc" in args.arch:
        mask_stack_list_list = [0 for _ in gflops_list]
    else:
        mask_stack_list_list = [[] for _ in gflops_list]
    upb_batch_gflops_list = []
    real_batch_gflops_list = []

    losses_dict = {}
    # YD: Initialize policy stats variables
    policy_layer_keep_count = np.zeros(shape=(len(mask_stack_list_list),), dtype=float)
    policy_layer_reuse_count = np.zeros(shape=(len(mask_stack_list_list),), dtype=float)
    policy_layer_skip_count = np.zeros(shape=(len(mask_stack_list_list),), dtype=float)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            # input and target
            batchsize = input_tuple[0].size(0)

            if args.gpus is None:
                input_data = input_tuple[0].cpu()

                target = input_tuple[-1].cpu()
            else:
                input_data = input_tuple[0].cuda(non_blocking=True)

                target = input_tuple[-1].cuda(non_blocking=True)

            # model forward function
            if 'dynstc' in args.arch:
                output, mask_stack_list, _, gate_meta, spatial_masks_list, dyn_outputs = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0)
            else:
                output, mask_stack_list, _, gate_meta = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0)
                spatial_masks_list = None

            # # YD TEMP DEBUG:
            # my_real_flops_per_frame_and_res_layer = torch.sum(dyn_outputs['flops_real'][0], dim=(0,1)).numpy()/(16*8)
            # for i in range(8):
            #     print('MY: ResNetLayer %d: FLOPS per frame: orig - %d, real - %d' %
            #           (i+1, dyn_outputs['flops_ori'][0][i].numpy()/(8), my_real_flops_per_frame_and_res_layer[i]))

            # measure losses, accuracy and predictions
            # TODO YD : This is kept for reference until replaced by "online" flops calculations
            upb_gflops_tensor, real_gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops,
                                                                           gflops_list, g_meta, args)
            # # YD TEMP:
            # # static flops: resnet blocks static flops + 1st conv flops + fully connected flops + downsampling flops
            # my_static_flops = (torch.sum(dyn_outputs['flops_ori']/8) + dyn_outputs['flops_real'][1]) / 10**9
            # my_real_flops = (torch.sum(dyn_outputs['flops_real'][0])/(16*8) + dyn_outputs['flops_real'][1]
            #                  + torch.sum(dyn_outputs["flops_downsample"])/8 +
            #                  torch.sum(dyn_outputs["flops_channels_mask"])/8) / 10 ** 9
            # print('ref base model: %f, my base model: %f' % (base_model_gflops, my_static_flops))
            # print('ref real flops: %f, my real flops: %f' % (real_gflops_tensor, my_real_flops))
            # # TODO YD: remaining differences are due to static model flops diff - probably batch norms not accounted for in my imp.
            #  YD TEMP DEBUG print:
            if i % 20 == 0:
                my_static_flops = (torch.sum(dyn_outputs['flops_ori'] / 8) + dyn_outputs['flops_real'][1]) / 10 ** 9
                my_real_flops = (torch.sum(dyn_outputs['flops_real'][0]) / (16 * 8) + dyn_outputs['flops_real'][1]
                                                  + torch.sum(dyn_outputs["flops_downsample"])/8 +
                                                  torch.sum(dyn_outputs["flops_channels_mask"])/8) / 10 ** 9
                my_upperbound_flops = (torch.sum(dyn_outputs['flops_upperbound']) / (16 * 8) + dyn_outputs['flops_real'][1]
                                                  + torch.sum(dyn_outputs["flops_downsample"])/8 +
                                                  torch.sum(dyn_outputs["flops_channels_mask"])/8) / 10 ** 9
                const_static_diff = base_model_gflops - my_static_flops
                real_flops_fixed = my_real_flops + const_static_diff
                upb_flops_fixed = my_upperbound_flops + const_static_diff
                print('batch %d: real flops (ref,my): (%f, %f), upb flops (ref,my): (%f, %f)' %
                      (i, real_gflops_tensor, real_flops_fixed, upb_gflops_tensor, upb_flops_fixed))

            loss_dict = compute_losses(criterion, output, target, mask_stack_list,
                                       upb_gflops_tensor, real_gflops_tensor, epoch, model,
                                       base_model_gflops, args)

            # TODO YD: make the following loss the criterion for optimization
            # TODO YD: Set real p in training
            p = 0.1
            closs, sloss, bloss = my_criterion(output, target[:, 0], dyn_outputs['flops_real'], dyn_outputs['flops_mask'],
                                               dyn_outputs['flops_ori'], dyn_outputs['flops_downsample'], args.batch_size,
                                               args.den_target, args.spatial_lambda, dyn_outputs['norm_s'],
                                               None, dyn_outputs['norm_s_t'], None, args.spatial_gamma, p)

            upb_batch_gflops_list.append(upb_gflops_tensor)
            real_batch_gflops_list.append(real_gflops_tensor)

            # YD: Collect debug & analysis data from batch
            policy_layer_keep_count, policy_layer_reuse_count, policy_layer_skip_count = \
                policy_stats.update_layers_policy_stats(mask_stack_list, policy_layer_keep_count,
                                                        policy_layer_reuse_count, policy_layer_skip_count)
            clips_class_ind = target.detach().cpu().numpy()[:, 0]
            if args.debug_frames_dir_path:#and (i % 50) == 0:
                vis_utils.print_videos_frames(args.debug_frames_dir_path, 'batch_%04d' % i,
                                              input_data.detach().cpu().numpy(),
                                              model.module.input_mean, model.module.input_std,
                                              class_names, clips_class_ind, args.debug_frames_spatial_masks,
                                              spatial_masks_list)

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            all_results.append(output)
            all_targets.append(target)

            # record loss and accuracy
            if len(losses_dict) == 0:
                losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
            for loss_name in loss_dict:
                losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
            top1.update(prec1.item(), batchsize)
            top5.update(prec5.item(), batchsize)

            # gather masks
            for layer_i, mask_stack in enumerate(mask_stack_list):
                if "batenet" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                # TODO YD: update mask list to include spatial mask
                elif "dynstc" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                elif "AdaBNInc" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                else:
                    raise ValueError('unsupported arch')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.
                                format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses_dict["loss"], top1=top1, top5=top5))

                for loss_name in losses_dict:
                    if loss_name == "loss" or "mask" in loss_name:
                        continue
                    print_output += ' {header:s} {loss.val:.3f}({loss.avg:.3f})'. \
                        format(header=loss_name[0], loss=losses_dict[loss_name])
                #print(print_output)
                # YD: print policy keep/reuse/skip stats
                total_policy_keep = np.sum(policy_layer_keep_count)
                total_policy_reuse = np.sum(policy_layer_reuse_count)
                total_policy_skip = np.sum(policy_layer_skip_count)
                total_policy_channels_num = total_policy_keep + total_policy_reuse + total_policy_skip
                print_output += ('  [Keep / Reuse / Skip]: [%4.2f%% / %4.2f%% / %4.2f%%]' %
                      (total_policy_keep / total_policy_channels_num * 100,
                       total_policy_reuse / total_policy_channels_num * 100,
                       total_policy_skip / total_policy_channels_num * 100))
                print(print_output)

    upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
    real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # multi-label mAP

    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses_dict["loss"]))

    usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
    print(usage_str)

    return mAP, mmAP, top1.avg, top5.avg, usage_str


def set_random_seed(the_seed, args):
    np.random.seed(the_seed)
    torch.manual_seed(the_seed)


def compute_losses(criterion, prediction, target, mask_stack_list, upb_gflops_tensor, real_gflops_tensor, epoch_i,
                   model,
                   base_model_gflops, args):
    loss_dict = {}
    if args.gflops_loss_type == "real":
        gflops_tensor = real_gflops_tensor
    else:
        gflops_tensor = upb_gflops_tensor

    # accuracy loss
    acc_loss = criterion(prediction, target[:, 0])
    loss_dict["acc_loss"] = acc_loss
    loss_dict["eff_loss"] = acc_loss * 0

    # gflops loss
    gflops_loss = acc_loss * 0
    if args.gate_gflops_loss_weight > 0 and epoch_i > args.eff_loss_after:
        if args.gflops_loss_norm == 1:
            gflops_loss = torch.abs(gflops_tensor - args.gate_gflops_bias) * args.gate_gflops_loss_weight
        elif args.gflops_loss_norm == 2:
            gflops_loss = ((
                                   gflops_tensor / base_model_gflops - args.gate_gflops_threshold) ** 2) * args.gate_gflops_loss_weight
        loss_dict["gflops_loss"] = gflops_loss
        loss_dict["eff_loss"] += gflops_loss

    # threshold loss for cgnet
    thres_loss = acc_loss * 0

    loss = acc_loss + gflops_loss + thres_loss
    loss_dict["loss"] = loss

    return loss_dict


def get_policy_usage_str(upb_gflops, real_gflops):
    return "Equivalent GFLOPS: upb: %.4f   real: %.4f" % (upb_gflops.item(), real_gflops.item())


def get_recorders(number):
    return [Recorder() for _ in range(number)]


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def setup_log_directory(exp_header, test_mode, args, logger):
    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    if test_mode:
        exp_full_path = ospj(common.EXPS_PATH, args.test_from)
    else:
        exp_full_path = ospj(common.EXPS_PATH, exp_full_name)

        os.makedirs(exp_full_path)
        os.makedirs(ospj(exp_full_path, "models"))
    if args.skip_log == False:
        logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


def main(argv):
    t0 = time.time()
    inner_main(argv)
    print("Finished in %.4f seconds\n" % (time.time() - t0))

if __name__ == "__main__":
    main(sys.argv[1:])

# # 1. Jester, TSN-Resnet18 + Spatial Mask (DGnet pretrain)
# -d
# gitlab-srv:4567/ishay/cbi:ub18_cu100_py376_pytorch120
# -C
# execute
# -M
# dynstc_inference_test.py
# -W
# /home/yonatand/cbi/AdaFuse
# -q
# gpu_inference_q
# -n
# 8
# -s
# 4gb
# -v
# /algo/CBI_artifacts:/algo/CBI_artifacts
# -v
# /algo/CBI_inputs:/algo/CBI_inputs
# -A
# "jester RGB --arch res18_dynstc_net --num_segments 8 --lr 0.002 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 1024 --gbn --grelu --gate_gflops_loss_weight 0.1 --gflops_loss_type upb --batch-size 16 -j 8 --gpus 0 --exp_header X --test_from g1213-145737_jester_8_bate18_1024_gsmx_g.1_tsn_upbg_b16_lr.002/models/ckpt.best.pth.tar --skip_log --static_flops_table_path /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_static_flops.csv --model_desc_table_path /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_batch16_model_desc.csv --spatial_masking --dgnet_pretrained_path /algo/CBI_artifacts/DGnet_ref_models/resdg18_05.pth.tar"

# # 2. SomethingV2, TSN-Resnet18 + Spatial Mask (DGnet pretrain)
# -d
# gitlab-srv:4567/ishay/cbi:ub18_cu100_py376_pytorch120
# -C
# execute
# -M
# dynstc_inference_test.py
# -W
# /home/yonatand/cbi/AdaFuse
# -q
# gpu_inference_q
# -n
# 8
# -s
# 4gb
# -v
# /algo/CBI_artifacts:/algo/CBI_artifacts
# -v
# /algo/CBI_inputs:/algo/CBI_inputs
# -A
# "somethingv2 RGB --arch res18_dynstc_net --num_segments 8 --lr 0.002 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 1024 --gbn --grelu --gate_gflops_loss_weight 0.1 --gflops_loss_type upb --batch-size 16 -j 8 --gpus 0 --exp_header X --test_from g1213-090449_sthv2_8_bate18_1024_gsmx_g.1_tsn_upbg_b16_lr.002/models/ckpt.best.pth.tar --skip_log --static_flops_table_path /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_static_flops.csv --model_desc_table_path /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_batch16_model_desc.csv --spatial_masking --dgnet_pretrained_path /algo/CBI_artifacts/DGnet_ref_models/resdg18_05.pth.tar"

# # 3. Local SomethingV2, TSN-Resnet18 + Spatial Mask (DGnet pretrain) + Debug masks
# somethingv2
# RGB
# --arch
# res18_dynstc_net
# --num_segments
# 8
# --lr
# 0.002
# --lr_steps
# 20
# 40
# --epochs
# 50
# --wd
# 5e-4
# --npb
# --ada_reso_skip
# --init_tau
# 0.67
# --gsmx
# --gate_history
# --gate_hidden_dim
# 1024
# --gbn
# --grelu
# --gate_gflops_loss_weight
# 0.1
# --gflops_loss_type
# upb
# --batch-size
# 16
# -j
# 0
# --exp_header
# X
# --test_from
# g1213-090449_sthv2_8_bate18_1024_gsmx_g.1_tsn_upbg_b16_lr.002/models/ckpt.best.pth.tar
# --skip_log
# --static_flops_table_path
# /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_static_flops.csv
# --model_desc_table_path
# /algo/CBI_artifacts/analysis/resnet18_dynstc_tsn_batch16_model_desc.csv
# --spatial_masking
# --debug_frames_dir_path
# /algo/CBI_artifacts/analysis/resnet18_tsn_somethingv2_dyn_stc_masks
# --debug_frames_spatial_masks
# --dgnet_pretrained_path
# /algo/CBI_artifacts/DGnet_ref_models/resdg18_04.pth.tar