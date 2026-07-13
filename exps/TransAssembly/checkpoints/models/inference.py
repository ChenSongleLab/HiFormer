
from .func import (linear_assignment,
                   get_trans_l2_loss, get_rot_l2_loss,
                   get_rot_cd_loss, get_shape_cd_loss_default,
                   get_total_cd_loss, get_contact_point_loss,
                   get_contact_point_loss_for_single_part)


def inference(preds, gt_poses, part_pcs, part_valids, match_ids, contact_points, sym_info, args=None):
    num_trans, batch_size, num_part, dim_pred = preds[0].size()

    pred_poses_per_trans = preds[0][num_trans - 1]


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
    shape_cd_loss_per_trans, dist1, dist2 = get_shape_cd_loss_default(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids,return_raw=True)
    total_cd_loss_per_trans, acc = get_total_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)
    contact_point_loss_per_trans, num_correct_contact, num_contact_points = get_contact_point_loss(pred_trans, pred_rot, contact_points, sym_info)

    acc = acc.sum(-1).float()
    return pred_poses_per_trans, total_cd_loss_per_trans, shape_cd_loss_per_trans, contact_point_loss_per_trans, acc, num_correct_contact, num_contact_points, dist1, dist2


def decode_eval(pred_poses, gt_poses, part_pcs, part_valids, match_ids, contact_points, sym_info, part_mask, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()

    pred_poses_per_trans = pred_poses[:, num_trans - 1]


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


    pred_poses_per_trans[part_mask.unsqueeze(-1).repeat(1, 1, 7)] = gt_poses[part_mask.unsqueeze(-1).repeat(1, 1, 7)]
    pred_trans = pred_poses_per_trans[:, :, :3]
    pred_rot = pred_poses_per_trans[:, :, 3:]
    gt_trans = gt_poses[:, :, :3]
    gt_rot = gt_poses[:, :, 3:]


    shape_dist_1, shape_dist_2 = get_shape_cd_loss_default(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids, return_raw=True)
    shape_valid = (~part_mask).unsqueeze(-1).repeat(1, 1, part_pcs.size(2)).view(batch_size, -1).contiguous()
    shape_dist_1 = shape_dist_1[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
    shape_dist_2 = shape_dist_2[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
    shape_cd_loss = shape_dist_1.mean(1) + shape_dist_2.mean(1)


    part_cd_loss, acc = get_total_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)
    acc = acc[~part_mask]

    contact_point_loss, num_correct_contact, num_contact_points = \
        get_contact_point_loss_for_single_part(pred_trans, pred_rot, contact_points, sym_info, part_mask=part_mask)

    return part_cd_loss, shape_cd_loss, contact_point_loss, acc, num_correct_contact, num_contact_points
