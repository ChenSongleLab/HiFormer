
import torch
import torch.nn.functional as F
from .box_ops import *
from .func import *
import numpy as np

def comp_group_losses(preds, gt_poses, part_pcs, part_valids, match_ids, args=None, **kwargs):
    num_trans, batch_size,  num_part, dim_pred = preds[0].size()
    pred_poses = preds[0]





    part_ids = kwargs["part_ids"]
    num_group = kwargs["num_group"]
    num_gpart = kwargs["num_group_part"]
    parts_valid_in_group = kwargs["parts_valid_in_group"]
    groups_valid = kwargs["groups_valid"]
    flat2group_index = kwargs["flat2group_index"]
    group2flat_index = kwargs["group2flat_index"]


    flat2group_index_1d = flat2group_index[:, :num_group, :num_gpart].reshape(batch_size, num_group * num_gpart)
    part_valids_group_mask = parts_valid_in_group.unsqueeze(-1).repeat(1, 1, 1, 3)

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[trans_ind]





        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = list()
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break
                cur_pts = part_pcs[bs_ind, need_to_match_part]


                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_centers = cut_gt_poses[:, :3]
                cur_gt_quats = cut_gt_poses[:, 3:]



                matched_pred_ids, matched_gt_ids = linear_assignment(cur_pts, cur_pred_centers, cur_pred_quats,
                                                                     cur_gt_centers, cur_gt_quats)
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]



        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)

        box_loss_per_trans = torch.zeros_like(trans_l2_loss_per_trans)
        group_loss_per_trans = torch.zeros_like(trans_l2_loss_per_trans)









        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()
        box_loss = box_loss_per_trans.mean()
        group_cd_loss = group_loss_per_trans.mean()


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd + \
                         box_loss * args.loss_weight_box +\
                         group_cd_loss * args.loss_weight_group_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_box_loss = box_loss
            total_group_cd_loss = group_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd + \
                          box_loss * args.loss_weight_box +\
                          group_cd_loss * args.loss_weight_group_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_box_loss += box_loss
            total_group_cd_loss += group_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_box_loss /= num_trans
    total_group_cd_loss /= num_trans
    total_part_vertex_l12_loss = torch.zeros_like(total_trans_l2_loss)
    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_box_loss, total_group_cd_loss, total_part_vertex_l12_loss


def comp_group_losses_fast(preds, gt_poses, part_pcs, part_valids, match_ids, args=None, **kwargs):
    num_trans, batch_size,  num_part, dim_pred = preds[0].size()
    num_point = part_pcs.size(2)
    pred_poses = preds[0]
    parts_box = preds[1]
    groups_box = preds[2]
    shape_box = preds[3]


    part_ids = kwargs["part_ids"]
    num_group = kwargs["num_group"]
    num_gpart = kwargs["num_group_part"]
    parts_valid_in_group = kwargs["parts_valid_in_group"]
    groups_valid = kwargs["groups_valid"]
    flat2group_index = kwargs["flat2group_index"]
    group2flat_index = kwargs["group2flat_index"]
    part_boxes = kwargs["part_boxes"]


    group2flat_index_1d = group2flat_index[..., 0] * num_gpart + group2flat_index[..., 1]
    group2flat_index_exp = group2flat_index_1d.unsqueeze(-1).repeat(1, 1, 256)


    box_vertex = box_xyxy_to_vextex(box_cxcywh_to_xyxy(part_boxes))



    flat2group_index_1d = flat2group_index[:, :num_group, :num_gpart].reshape(batch_size, num_group * num_gpart)
    flat2group_index_points = flat2group_index_1d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_point,  3)
    flat2group_index_box_vertex = flat2group_index_1d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8,  3)
    flat2group_index_axis_align_box = flat2group_index_1d.unsqueeze(-1).repeat(1, 1, 6)
    flat2group_index_pose = flat2group_index_1d.unsqueeze(-1).repeat(1, 1, 7)

    part_valids_group_mask = parts_valid_in_group.unsqueeze(-1).repeat(1, 1, 1, 3)
    part_valids_group_points = parts_valid_in_group.unsqueeze(-1).repeat(1, 1, 1, num_point)
    part_valids_group_points_xyz = parts_valid_in_group.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_point, 3)
    part_valids_group_mask_fill = (part_valids_group_points_xyz - 1) * -10000
    part_valids_group_box_vertex = parts_valid_in_group.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 8, 3)
    part_valids_group_mask_box_vertex_fill = (part_valids_group_box_vertex - 1) * -10000



    gt_poses_gp = torch.gather(gt_poses, dim=1, index=flat2group_index_pose)


    gt_trans = gt_poses[:, :, :3]
    gt_rot = gt_poses[:, :, 3:]









    gt_center_vertex = gt_trans.unsqueeze(2).repeat(1, 1, 8, 1)
    gt_rot_vertex = gt_rot.unsqueeze(2).repeat(1, 1, 8, 1)
    gt_rot_box_vertex = qrot(gt_rot_vertex, box_vertex) + gt_center_vertex
    gt_rot_box_vertex_group = torch.gather(gt_rot_box_vertex, dim=1, index=flat2group_index_box_vertex)
    gt_rot_box_vertex_group = gt_rot_box_vertex_group.view(batch_size, num_group, num_gpart, 8, 3)
    gt_rot_box_vertex_group = gt_rot_box_vertex_group * part_valids_group_box_vertex
    gt_rot_box_vertex_group = gt_rot_box_vertex_group + part_valids_group_mask_box_vertex_fill



    for trans_ind in range(num_trans):

        parts_box_per_trans = parts_box[trans_ind]
        parts_box_per_trans_gp = torch.gather(parts_box_per_trans, dim=1, index=flat2group_index_axis_align_box)


        pred_poses_per_trans = pred_poses[trans_ind]
        pred_poses_per_trans_gp = torch.gather(pred_poses_per_trans, dim=1, index=flat2group_index_pose)

        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        pred_rot_rep = pred_rot.unsqueeze(2).repeat(1, 1, num_point, 1)
        pred_trans_rep = pred_trans.unsqueeze(2).repeat(1, 1, num_point, 1)
        pred_points = qrot(pred_rot_rep, part_pcs) + pred_trans_rep

        pred_part_points_group = torch.gather(pred_points, dim=1, index=flat2group_index_points)
        pred_part_points_group = pred_part_points_group.view(batch_size, num_group, num_gpart, num_point, 3)
        pred_part_points_group = pred_part_points_group * part_valids_group_points_xyz


        pred_center_vertex = pred_trans.unsqueeze(2).repeat(1, 1, 8, 1)
        pred_rot_vertex = pred_rot.unsqueeze(2).repeat(1, 1, 8, 1)
        pred_rot_box_vertex = qrot(pred_rot_vertex, box_vertex) + pred_center_vertex
        pred_rot_box_vertex_group = torch.gather(pred_rot_box_vertex, dim=1, index=flat2group_index_box_vertex)
        pred_rot_box_vertex_group = pred_rot_box_vertex_group.view(batch_size, num_group, num_gpart, 8, 3)
        pred_rot_box_vertex_group = pred_rot_box_vertex_group * part_valids_group_box_vertex
        pred_rot_box_vertex_group = pred_rot_box_vertex_group + part_valids_group_mask_box_vertex_fill


        with torch.no_grad():
            pred_rot_box_vertex_gp = pred_rot_box_vertex_group.view(batch_size, num_group, num_gpart, 24)
            pred_rot_box_vertex_gp = pred_rot_box_vertex_gp.unsqueeze(3).repeat(1, 1, 1, num_gpart, 1)
            gt_rot_box_vertex_gp = gt_rot_box_vertex_group.view(batch_size, num_group, num_gpart, 24)
            gt_rot_box_vertex_gp = gt_rot_box_vertex_gp.unsqueeze(2).repeat(1, 1, num_gpart, 1, 1)

            dist_mat = _get_axis_box_loss(pred_rot_box_vertex_gp, gt_rot_box_vertex_gp)
            dist_mat = dist_mat.view(batch_size*num_group, num_gpart,num_gpart).cpu()
            valid_in_gp_sum = parts_valid_in_group.view(batch_size*num_group, num_gpart).sum(-1)

            indices = [linear_sum_assignment(dist_mat[i])if valid_in_gp_sum[i] > 1 else (np.arange(0,num_gpart),np.arange(0,num_gpart)) for i in range(batch_size*num_group)]

            full_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            idx_pred = _get_src_permutation_idx(full_indices)
            idx_gt = _get_tgt_permutation_idx(full_indices)

        pred_poses_per_trans_gp_match = pred_poses_per_trans_gp.reshape(batch_size*num_group, num_gpart,7)[idx_pred]
        pred_poses_per_trans_gp_match = pred_poses_per_trans_gp_match.reshape(batch_size, num_group * num_gpart, 7)
        pred_poses_per_trans_match = torch.gather(pred_poses_per_trans_gp_match, dim=1, index=group2flat_index_exp[:,:,:7])

        gt_poses_gp_match = gt_poses_gp.reshape(batch_size*num_group, num_gpart,7)[idx_gt]
        gt_poses_gp_match = gt_poses_gp_match.reshape(batch_size, num_group * num_gpart, 7)
        gt_poses_match = torch.gather(gt_poses_gp_match, dim=1, index=group2flat_index_exp[:, :, :7])








        parts_box_per_trans_gp_match = parts_box_per_trans_gp.reshape(batch_size*num_group, num_gpart,6)[idx_pred]
        parts_box_per_trans_gp_match = parts_box_per_trans_gp_match.reshape(batch_size, num_group * num_gpart, 6)
        parts_box_per_trans_match = torch.gather(parts_box_per_trans_gp_match, dim=1, index=group2flat_index_exp[:, :, :6])





        groups_box_per_trans = groups_box[trans_ind]
        shape_box_per_trans = shape_box[trans_ind]

        pred_trans = pred_poses_per_trans_match[:, :, :3]
        pred_rot = pred_poses_per_trans_match[:, :, 3:]
        gt_trans = gt_poses_match[:, :, :3]
        gt_rot = gt_poses_match[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)



        if args.cascade_pred and trans_ind == 0:
            box_loss_per_trans = torch.zeros_like(trans_l2_loss_per_trans)
            group_loss_per_trans = torch.zeros_like(trans_l2_loss_per_trans)
        else:

            box_loss_per_trans = get_total_box_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids, flat2group_index_1d, \
                                                    part_valids_group_mask, groups_valid, shape_box_per_trans, groups_box_per_trans, parts_box_per_trans_match)


            group_loss_per_trans = get_group_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids, flat2group_index_1d, \
                                                    parts_valid_in_group, groups_valid)



        part_vertex_l12_loss_per_trans = torch.zeros_like(group_loss_per_trans)

        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()
        box_loss = box_loss_per_trans.mean()
        group_cd_loss = group_loss_per_trans.mean()
        part_vertex_l12_loss = part_vertex_l12_loss_per_trans.mean()


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd + \
                         box_loss * args.loss_weight_box +\
                         group_cd_loss * args.loss_weight_group_cd + \
                         part_vertex_l12_loss * args.loss_part_vertex_l12
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_box_loss = box_loss
            total_group_cd_loss = group_cd_loss
            total_part_vertex_l12_loss = part_vertex_l12_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd + \
                          box_loss * args.loss_weight_box +\
                          group_cd_loss * args.loss_weight_group_cd + \
                          part_vertex_l12_loss * args.loss_part_vertex_l12
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_box_loss += box_loss
            total_group_cd_loss += group_cd_loss
            total_part_vertex_l12_loss += part_vertex_l12_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_box_loss /= num_trans
    total_group_cd_loss /= num_trans
    total_part_vertex_l12_loss /= num_trans

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_box_loss, total_group_cd_loss, total_part_vertex_l12_loss

def _get_axis_box_loss(gt_vertexs, pred_vertexs):
    pred_vertexs = pred_vertexs.view(-1, 8, 3)
    box_part_pred_min = pred_vertexs.min(1)[0]
    box_part_pred_max = pred_vertexs.max(1)[0]
    box_part_pred = torch.cat([(box_part_pred_min + box_part_pred_max)/2, (box_part_pred_max - box_part_pred_min).clamp(min=0.01)],dim=-1)

    gt_vertexs = gt_vertexs.view(-1, 8, 3)
    box_part_tgt_min = gt_vertexs.min(1)[0]
    box_part_tgt_max = gt_vertexs.max(1)[0]
    box_part_tgt =  torch.cat([(box_part_tgt_min + box_part_tgt_max)/2, (box_part_tgt_max - box_part_tgt_min).clamp(min=0.01)], dim=-1)
    loss_axis_box = get_loss_box(box_part_pred, box_part_tgt)
    return loss_axis_box

def _get_src_permutation_idx(indices):

    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):

    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def comp_losses(preds, gt_poses, part_pcs, part_valids, match_ids, args=None):
    num_trans, batch_size,  num_part, dim_pred = preds[0].size()
    pred_poses = preds[0]
    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[trans_ind]


        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = list()
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break
                cur_pts = part_pcs[bs_ind, need_to_match_part]


                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_centers = cut_gt_poses[:, :3]
                cur_gt_quats = cut_gt_poses[:, 3:]
                cur_pred_box = parts_box_per_trans[bs_ind, need_to_match_part]


                matched_pred_ids, matched_gt_ids = linear_assignment(cur_pts, cur_pred_centers, cur_pred_quats,
                                                                     cur_gt_centers, cur_gt_quats)
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]


        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)


        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss

def comp_decoder_losses(pred_poses, gt_poses, pred_cates, gt_cates, part_pcs, pos_ids, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    _, num_pos = pos_ids.size()
    pred_poses = pred_poses[:, :, :num_pos, :]
    part_pcs = part_pcs[:, :num_pos].contiguous()
    part_valids = pred_poses.new_ones((batch_size, num_pos))

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]
        pred_cates_per_trans = pred_cates[:, trans_ind]


        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)


        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()


        cate_loss = F.binary_cross_entropy_with_logits(pred_cates_per_trans, gt_cates, reduction="mean")


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd + \
                         cate_loss * args.loss_weight_cate
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_cate_loss = cate_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd + \
                          cate_loss * args.loss_weight_cate
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_cate_loss += cate_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_cate_loss /= num_trans

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_cate_loss


def comp_decoder_losses_v2(pred_poses, gt_poses, part_pcs, pos_ids, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    _, num_pos = pos_ids.size()
    pred_poses = pred_poses[:, :, :num_pos, :]
    part_pcs = part_pcs[:, :num_pos].contiguous()
    part_valids = pred_poses.new_ones((batch_size, num_pos))

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]


        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)


        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss


def comp_decoder_losses_freeze(pred_poses, gt_poses, part_pcs, part_valids, match_ids, part_mask, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]


        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = list()
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break
                cur_pts = part_pcs[bs_ind, need_to_match_part]


                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_centers = cut_gt_poses[:, :3]
                cur_gt_quats = cut_gt_poses[:, 3:]


                matched_pred_ids, matched_gt_ids = linear_assignment(cur_pts, cur_pred_centers, cur_pred_quats,
                                                                     cur_gt_centers, cur_gt_quats)
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]



        part_mask_3 = part_mask.transpose(0, 1).repeat(1, 1, dim_pred)
        pred_poses_per_trans[part_mask_3] = gt_poses[part_mask_3]
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]

        part_mask_2 = (~part_mask).transpose(0, 1).squeeze(-1)

        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids, return_raw=True)[part_mask_2]


        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids, return_raw=True)[part_mask_2]

        mask_valid = part_mask_2.unsqueeze(-1).repeat(1, 1, part_pcs.size(2))

        rot_dist_1, rot_dist_2 = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids, return_raw=True)
        rot_valid = mask_valid.view(-1, part_pcs.size(2)).contiguous()
        rot_dist_1 = rot_dist_1[rot_valid].view(batch_size, part_pcs.size(2)).contiguous()
        rot_dist_2 = rot_dist_2[rot_valid].view(batch_size, part_pcs.size(2)).contiguous()
        rot_cd_loss_per_trans = rot_dist_1.mean(1) + rot_dist_2.mean(1)


        shape_dist_1, shape_dist_2 = get_shape_cd_loss(part_pcs, pred_rot, gt_rot,
                                                       pred_trans, gt_trans, part_valids, return_raw=True)
        shape_valid = mask_valid.view(batch_size, -1).contiguous()
        shape_dist_1 = shape_dist_1[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
        shape_dist_2 = shape_dist_2[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
        shape_cd_loss_per_trans = shape_dist_1.mean(1) + shape_dist_2.mean(1)


        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()


        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss

