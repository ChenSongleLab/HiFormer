
import time
import copy
import torch
from scripts.z_utils import EtaMeter, AverageMeter, ProgressMeter, adjust_learning_rate, adjust_moco_momentum
from datasets.partnet import DATA_FEATURES as data_features
from models.losses import comp_losses
import numpy as np
from models.trans_assembly_encoder import show_process_gpu_info

def train_one_epoch(train_loader, model, optimizer, scaler, summary_writer, log_writer, epoch, countdown, args):
    eta_time = EtaMeter('Eta')
    batch_time = AverageMeter('Time', ':6.6f')
    data_time = AverageMeter('Data', ':6.6f')

    learning_rates = AverageMeter('lr', ':6.6f')
    learning_rates_backbone = AverageMeter('lr_bk', ':6.6f')


    losses = AverageMeter('TotalLoss', ':6.6f')
    trans_l2_losses = AverageMeter('TransL2Loss', ':6.6f')
    rot_l2_losses = AverageMeter('RotL2Loss', ':6.6f')
    rot_cd_losses = AverageMeter('RotCDLoss', ':6.6f')
    shape_cd_losses = AverageMeter('ShapeCDLoss', ':6.6f')
    pointnet_losses = AverageMeter('PointNetLoss', ':6.6f')
    box_losses = AverageMeter('BoxLoss', ':6.6f')
    group_cd_losses = AverageMeter('groupCDloss', ':6.6f')

    progress = ProgressMeter(
        len(train_loader),
        [eta_time, batch_time, data_time,
         learning_rates, learning_rates_backbone, losses,
         trans_l2_losses, rot_l2_losses,
         rot_cd_losses, shape_cd_losses,
         pointnet_losses,
         box_losses,
         group_cd_losses],
        prefix="Epoch: [{}]".format(epoch + 1))

    if args.decode_on and not args.encode_freeze:
        decode_losses = AverageMeter('DecodeLoss', ':6.6f')
        progress.append(decode_losses)
        if args.cate_on:
            decode_cate_losses = AverageMeter('DecodeCateLoss', ':6.6f')
            progress.append(decode_cate_losses)


    model.train()

    epoch_start_time = end = time.time()
    iters_per_epoch = len(train_loader)
    for i, batch_data in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.type_sched == "cosine":
            lr, lr_backbone = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(optimizer.param_groups[0]["lr"])



        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0)
        part_valids = torch.cat(batch_data[data_features.index('part_valids')], dim=0)
        gt_part_poses = torch.cat(batch_data[data_features.index('part_poses')], dim=0)
        match_ids = batch_data[data_features.index('match_ids')]
        part_ids = torch.cat(batch_data[data_features.index('part_ids')], dim=0)
        shape_id = batch_data[data_features.index('shape_id')]
        part_boxes = torch.cat(batch_data[data_features.index('input_part_box')], dim=0)



        parts_valid_in_group = torch.cat(batch_data[data_features.index('parts_valid_in_group')], dim=0)
        groups_valid = torch.cat(batch_data[data_features.index('groups_valid')], dim=0)
        flat2group_index = torch.cat(batch_data[data_features.index('flat2group_index')], dim=0)
        group2flat_index = torch.cat(batch_data[data_features.index('group2flat_index')], dim=0)
        num_group, num_group_part, parts_groups_index, memory_block_need = split_group_data(part_valids, part_ids)
        parts_valid_in_group = parts_valid_in_group[:,:num_group, :num_group_part,...]
        groups_valid = groups_valid[:,:num_group]



        start_time = time.time()
        if args.gpu is not None:
            part_pcs = part_pcs.cuda(args.gpu, non_blocking=True)
            part_valids = part_valids.cuda(args.gpu, non_blocking=True)
            gt_part_poses = gt_part_poses.cuda(args.gpu, non_blocking=True)
            part_ids = part_ids.cuda(args.gpu, non_blocking=True)
            part_boxes = part_boxes.cuda(args.gpu, non_blocking=True)


            parts_valid_in_group = parts_valid_in_group.cuda(args.gpu, non_blocking=True)
            groups_valid = groups_valid.cuda(args.gpu, non_blocking=True)
            flat2group_index = flat2group_index.cuda(args.gpu, non_blocking=True)
            group2flat_index = group2flat_index.cuda(args.gpu, non_blocking=True)




        pred_part_poses, loss, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss, pointnet_loss, box_loss, group_cd_loss, output \
            = model(shape_id, part_pcs, part_boxes, part_valids, gt_part_poses, match_ids, part_ids, num_group, num_group_part, parts_valid_in_group, groups_valid, flat2group_index, group2flat_index)
        forward_time = time.time()-start_time
        start_time = time.time()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = part_pcs.size(0)

        losses.update(loss.item(), batch_size)
        trans_l2_losses.update(trans_l2_loss.item(), batch_size)
        rot_l2_losses.update(rot_l2_loss.item(), batch_size)
        rot_cd_losses.update(rot_cd_loss.item(), batch_size)
        shape_cd_losses.update(shape_cd_loss, batch_size)
        pointnet_losses.update(pointnet_loss, batch_size)
        box_losses.update(box_loss, batch_size)
        group_cd_losses.update(group_cd_loss, batch_size)

        if args.decode_on and not args.encode_freeze:
            decode_losses.update(output["decode_loss"], batch_size)
            if args.cate_on:
                decode_cate_losses.update(output["decode_cate_loss"], batch_size)

        if args.rank == 0:
            summary_writer.add_scalar("total_loss", loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("total_loss_per_epoch", losses.avg, epoch + 1)
            summary_writer.add_scalar("trans_l2_loss", trans_l2_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("trans_l2_loss_per_epoch", trans_l2_losses.avg, epoch + 1)
            summary_writer.add_scalar("rot_l2_loss", rot_l2_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("rot_l2_loss_per_epoch", rot_l2_losses.avg, epoch + 1)
            summary_writer.add_scalar("rot_cd_loss", rot_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("rot_cd_loss_per_epoch", rot_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("shape_cd_loss", shape_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("shape_cd_loss_per_epoch", shape_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("pointnet_loss", pointnet_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("pointnet_loss_per_epoch", pointnet_losses.avg, epoch + 1)
            summary_writer.add_scalar("box_loss_per_epoch", box_losses.avg, epoch + 1)
            summary_writer.add_scalar("group_cd_loss_per_epoch", group_cd_losses.avg, epoch + 1)



        batch_time.update(time.time() - end)
        end = time.time()


        eta_time.update(batch_time.avg * (len(train_loader) - i))
        ep_countdown = batch_time.avg if countdown == 0 else countdown
        eta_time.update_ep(ep_countdown * (len(train_loader) * (args.epochs - epoch - 1) + len(train_loader) - i))

        if i % args.print_freq == 0:
            infos = progress.display(i)
            log_writer.write(infos + "\n")
            log_writer.flush()

    return batch_time.avg

def split_group_data(part_valids, part_ids):

    batch_size, num_part = part_valids.shape
    parts_groups_index = []
    num_group = 0
    num_group_part = 0
    for b in range(batch_size):
        cur_same_class_dict = {}
        for j in range(int(part_valids[b].sum())):
            cur_class = int(part_ids[b][j])
            if cur_class not in cur_same_class_dict:
                cur_same_class_dict[int(cur_class)] = []
            cur_same_class_dict[int(cur_class)].append(j)
            num_group_part = max(num_group_part, len(cur_same_class_dict[int(cur_class)]))
        num_group = max(num_group, len(cur_same_class_dict))
        parts_groups_index.append(cur_same_class_dict)
    memory_block_need = batch_size * num_group * num_group_part
    return num_group, num_group_part, parts_groups_index, memory_block_need