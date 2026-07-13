
import os.path
import time
import copy
import torch
from scripts.z_utils import AverageMeter, ProgressMeter
from datasets.partnet import DATA_FEATURES as data_features

from models.trans_assembly_encoder import TransAssembly_encoder,show_process_gpu_info

def eval_func(val_loader, model, log_writer, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')

    shape_chamfer_dist = AverageMeter('Shape Chamfer Distance', ':6.4f')
    part_acc = AverageMeter('Part Accuracy', ':6.4f')
    connectivity_acc = AverageMeter('Connectivity Accuracy', ':6.4f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time,
         shape_chamfer_dist, part_acc, connectivity_acc
         ],
        prefix="TransAssembly Inference: ")


    model.eval()

    end = time.time()
    val_num_batch = len(val_loader)

    sum_part_cd_loss = 0.
    sum_shape_cd_loss = 0.
    sum_contact_point_loss = 0.
    total_acc_part = 0.
    total_valid_part = 0.
    total_contact_correct = 0.
    total_contact_point = 0.
    num_ins = 0
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)


        if not batch_data:
            continue


        if args.gt_vis:
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in args.gt_vis_dir:
                args.gt_vis_dir = os.path.join(root_dir, args.gt_vis_dir)

            part_pcs = batch_data[data_features.index('part_pcs')]
            part_valids = batch_data[data_features.index('part_valids')]
            part_poses = batch_data[data_features.index('part_poses')]
            cur_batch_size = gt_vis(part_pcs, part_poses, part_valids, num_ins, args=args)
            num_ins += cur_batch_size
            continue


        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0)
        part_valids = torch.cat(batch_data[data_features.index('part_valids')], dim=0)
        gt_part_poses = torch.cat(batch_data[data_features.index('part_poses')], dim=0)
        match_ids = batch_data[data_features.index('match_ids')]
        part_ids = torch.cat(batch_data[data_features.index('part_ids')], dim=0)
        contact_points = torch.cat(batch_data[data_features.index('contact_points')], dim=0)
        sym_info = torch.cat(batch_data[data_features.index('sym')], dim=0)
        shape_id = batch_data[data_features.index('shape_id')]
        part_boxes = torch.cat(batch_data[data_features.index('input_part_box')], dim=0)


        parts_valid_in_group = torch.cat(batch_data[data_features.index('parts_valid_in_group')], dim=0)
        groups_valid = torch.cat(batch_data[data_features.index('groups_valid')], dim=0)
        flat2group_index = torch.cat(batch_data[data_features.index('flat2group_index')], dim=0)
        group2flat_index = torch.cat(batch_data[data_features.index('group2flat_index')], dim=0)
        num_group, num_group_part, parts_groups_index, memory_block_need = split_group_data(part_valids, part_ids)
        parts_valid_in_group = parts_valid_in_group[:, :num_group, :num_group_part, ...]
        groups_valid = groups_valid[:, :num_group]

        if args.gpu is not None:
            part_pcs = part_pcs.cuda(args.gpu, non_blocking=True)
            part_valids = part_valids.cuda(args.gpu, non_blocking=True)
            gt_part_poses = gt_part_poses.cuda(args.gpu, non_blocking=True)
            part_ids = part_ids.cuda(args.gpu, non_blocking=True)
            part_boxes = part_boxes.cuda(args.gpu, non_blocking=True)
            contact_points = contact_points.cuda(args.gpu, non_blocking=True)
            sym_info = sym_info.cuda(args.gpu, non_blocking=True)


            parts_valid_in_group = parts_valid_in_group.cuda(args.gpu, non_blocking=True)
            groups_valid = groups_valid.cuda(args.gpu, non_blocking=True)
            flat2group_index = flat2group_index.cuda(args.gpu, non_blocking=True)
            group2flat_index = group2flat_index.cuda(args.gpu, non_blocking=True)


        with torch.no_grad():
            part_cd_loss, shape_cd_loss, contact_point_loss, acc_per_batch, valid_per_batch, \
                contact_correct_per_batch, contact_point_per_batch, batch_size, output \
                = model(shape_id, part_pcs, part_boxes, part_valids, gt_part_poses, match_ids, part_ids, num_group, \
                        num_group_part, parts_valid_in_group, groups_valid, flat2group_index, group2flat_index,\
                        contact_points=contact_points, sym_info=sym_info)


            if args.pred_encoder_vis:
                root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                if root_dir not in args.pred_encoder_vis_dir:
                    args.pred_encoder_vis_dir = os.path.join(root_dir, args.pred_encoder_vis_dir)

                pred_poses = output["pred_poses"]
                _ = pred_pose_vis(part_pcs, pred_poses, part_valids, num_ins, args=args)


            sum_part_cd_loss += part_cd_loss * batch_size
            sum_shape_cd_loss += shape_cd_loss * batch_size
            sum_contact_point_loss += contact_point_loss * batch_size

            total_acc_part += acc_per_batch
            total_valid_part += valid_per_batch
            total_contact_correct += contact_correct_per_batch
            total_contact_point += contact_point_per_batch
            num_ins += batch_size


        batch_time.update(time.time() - end)
        end = time.time()

        shape_chamfer_dist.update(shape_cd_loss.item(), batch_size)
        part_acc.update((acc_per_batch / valid_per_batch).item() * 100., batch_size)
        connectivity_acc.update((contact_correct_per_batch / contact_point_per_batch).item() * 100., batch_size)
        if i % args.print_freq == 0:
            infos = progress.display(i)

    if args.gt_vis:
        return None


    res_shape_cd = sum_shape_cd_loss / num_ins
    res_part_acc = total_acc_part / total_valid_part * 100.
    res_contact_acc = total_contact_correct / total_contact_point * 100.
    print("==========================================================")
    print("Shape Chamfer Distance: {}".format(res_shape_cd.item()))
    print("Part Accuracy: {}".format(res_part_acc.item()))
    print("Connectivity Accuracy: {}".format(res_contact_acc.item()))
    print("==========================================================")


    res_info = "\n==========================================================" + "\n" + \
               "Shape Chamfer Distance: {}".format(res_shape_cd.item()) + "\t" + \
               "Part Accuracy: {}".format(res_part_acc.item()) + "\t" + \
               "Connectivity Accuracy: {}".format(res_contact_acc.item()) + "\n" + \
               "==========================================================\n"
    log_writer.write(res_info + "\n")
    log_writer.flush()
    if args.rank == 0:
        summary_writer.add_scalar("eval_shape_chamfer_distance", res_shape_cd.item(), epoch + 1)
        summary_writer.add_scalar("eval_part_accuracy", res_part_acc.item(), epoch + 1)
        summary_writer.add_scalar("eval_connectivity_accuracy", res_contact_acc.item(), epoch + 1)

    return res_shape_cd.item(), res_part_acc.item(), res_contact_acc.item()

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