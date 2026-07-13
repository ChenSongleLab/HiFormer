
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
import torch
from cd.chamfer import chamfer_distance
from quaternion import qrot
from scipy.optimize import linear_sum_assignment
from .box_ops import *

def linear_assignment(pts, centers1, quats1, centers2, quats2):
    """
        Input: * x N x 3, * x 3, * x 4, * x 3, * x 4,
        Output: *, * (two lists)
    """
    import random
    pts_to_select = torch.tensor(random.sample([i for i in range(pts.size(1))], 100))
    pts = pts[:, pts_to_select]
    cur_part_cnt, num_point, _ = pts.size()

    with torch.no_grad():
        cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

        cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

        cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
        cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
        dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
        dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
        rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

    return rind, cind


def get_trans_l2_loss(trans1, trans2, valids, return_raw=False):
    """
        Input: B x P x 3, B x P x 3, B x P
        Output: B
    """
    loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data


def get_rot_l2_loss(pts, quat1, quat2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    num_point = pts.shape[2]

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data


def get_rot_cd_loss(pts, quat1, quat2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, _, num_point, _ = pts.size()

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)


    part_valids_points_xyz = valids.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_point, 3)
    part_valids_mask_fill = (part_valids_points_xyz - 1) * -10000
    pts1 = pts1 + part_valids_mask_fill
    pts2 = pts2 + part_valids_mask_fill


    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def get_shape_cd_loss(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2


    part_valids_points_xyz = valids.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_point, 3)
    part_valids_mask_fill = (part_valids_points_xyz - 1) * -10000
    pts1 = pts1 + part_valids_mask_fill
    pts2 = pts2 + part_valids_mask_fill


    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = (torch.sum(dist1, dim=1) + torch.sum(dist2, dim=1)) / torch.sum(valids, dim=1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data




def get_shape_cd_loss_default(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, num_part, num_point, _ = pts.size()
    valids_center = valids.unsqueeze(2).repeat(1, 1, 3)
    center1 = center1*valids_center
    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    if return_raw:
        return loss_per_data, dist1, dist2
    else:
        return loss_per_data


def get_total_cd_loss(pts, quat1, quat2, center1, center2, valids):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B, B x P
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)

    thresh = 0.01
    acc = (loss_per_data < thresh).float() * valids
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data, acc


def get_sym_point(point, x, y, z):
    if point.dim() == 1:
        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

    elif point.dim() == 2:
        if x:
            point[:, 0] = - point[:, 0]
        if y:
            point[:, 1] = - point[:, 1]
        if z:
            point[:, 2] = - point[:, 2]

    else:
        raise NotImplementedError

    return point.tolist()


def get_possible_point_list(point, sym=None):
    point_list = []
    sym = torch.tensor([1.0, 1.0, 1.0])
    if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
    elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
    elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
    elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 1, 1, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 0, 1))
    elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
    else:
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 1, 0))
        point_list.append(get_sym_point(point, 1, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
        point_list.append(get_sym_point(point, 1, 1, 1))
    return point_list


def get_min_l2_dist(list1, list2, center1, center2, quat1, quat2):
    num_part = list1.size(0)
    len1 = list1.size(1)
    len2 = list2.size(1)

    center1 = center1.unsqueeze(1).repeat(1, len1, 1)
    center2 = center2.unsqueeze(1).repeat(1, len2, 1)
    quat1 = quat1.unsqueeze(1).repeat(1, len1, 1)
    quat2 = quat2.unsqueeze(1).repeat(1, len2, 1)

    list1 = center1 + qrot(quat1, list1)
    list2 = center2 + qrot(quat2, list2)

    mat1 = list1.unsqueeze(2).repeat(1, 1, len2, 1)
    mat2 = list2.unsqueeze(1).repeat(1, len1, 1, 1)
    mat = (mat1 - mat2) * (mat1 - mat2)
    mat = mat.sum(dim=-1).view(num_part, -1)
    dist, _ = mat.min(-1)
    return dist


def get_contact_point_loss(center, quat, contact_points, sym_info):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    thresh = 0.01
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]
        contact_1 = (cur_contact_point[..., 0] == 1).view(-1)
        contact_2 = ((cur_contact_point[..., 0].transpose(0, 1).contiguous()) == 1).view(-1)
        if contact_1.sum() == 0:
            continue

        contact_point_1 = cur_contact_point.view(-1, 4)[contact_1][:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous().view(-1, 4)[contact_2][:, 1:]

        cur_sym = sym_info[bs_ind]
        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1).view(-1, 3)[contact_1]
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 3)[contact_2]

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1).view(-1, 4)[contact_1]
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 4)[contact_2]

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < thresh).sum()
        num_contact_pairs[bs_ind] = contact_1.sum()
        contact_point_loss[bs_ind] = dists.sum()

    return contact_point_loss, num_correct_pairs, num_contact_pairs


def get_contact_point_loss_for_single_part(center, quat, contact_points, sym_info, part_mask):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    thresh = 0.01
    _, pos_ids = (~part_mask).nonzero(as_tuple=True)
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]
        pos_id = pos_ids[bs_ind]

        contact = (cur_contact_point[pos_id][..., 0] == 1).view(-1)
        if contact.sum() == 0:
            continue
        contact_3 = contact.unsqueeze(-1).repeat(1, 3)
        contact_4 = contact.unsqueeze(-1).repeat(1, 4)

        contact_point_1 = cur_contact_point[pos_id][contact_4].view(-1, 4).contiguous()[:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous()[pos_id][contact_4].view(-1, 4).contiguous()[:, 1:]

        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_3].view(-1, 3).contiguous()
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_3].view(-1, 3).contiguous()

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_4].view(-1, 4).contiguous()
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_4].view(-1, 4).contiguous()

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < thresh).sum()
        num_contact_pairs[bs_ind] = contact.sum()
        contact_point_loss[bs_ind] = dists.sum()

    return contact_point_loss, num_correct_pairs, num_contact_pairs










def get_group_cd_loss(pts, quart_pred, quart_tgt, center_pred, center_tgt, valids, flat2group_index_1d, parts_valid_in_group, groups_valid):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B, B x P
    """
    batch_size, num_part, num_point, _ = pts.size()
    batch_size, num_group, num_gpart = parts_valid_in_group.size()


    center_pred = center_pred.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts_pred = qrot(quart_pred.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center_pred
    center_tgt = center_tgt.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts_tgt = qrot(quart_tgt.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center_tgt


    flat2group_index_points = flat2group_index_1d.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_point, 3)
    part_valids_group_points_xyz = parts_valid_in_group.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_point, 3)
    part_valids_group_mask_fill = (part_valids_group_points_xyz - 1) * -10000
    part_valids_group_points = parts_valid_in_group.unsqueeze(-1).repeat(1, 1, 1, num_point)



    gt_part_points_group = torch.gather(pts_tgt, dim=1, index=flat2group_index_points)
    gt_part_points_group = gt_part_points_group.view(batch_size, num_group, num_gpart, num_point, 3)
    gt_part_points_group = gt_part_points_group * part_valids_group_points_xyz
    gt_part_points_group = gt_part_points_group + part_valids_group_mask_fill
    gt_part_points_group = gt_part_points_group.view(batch_size*num_group, num_gpart*num_point, 3)


    pred_part_points_group = torch.gather(pts_pred, dim=1, index=flat2group_index_points)
    pred_part_points_group = pred_part_points_group.view(batch_size, num_group, num_gpart, num_point, 3)
    pred_part_points_group = pred_part_points_group * part_valids_group_points_xyz
    pred_part_points_group = pred_part_points_group + part_valids_group_mask_fill
    pred_part_points_group = pred_part_points_group.view(batch_size*num_group, num_gpart*num_point, 3)


    dist1, dist2 = chamfer_distance(gt_part_points_group, pred_part_points_group, transpose=False)
    valids = part_valids_group_points.view(batch_size*num_group, num_gpart*num_point)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_group_data = (torch.sum(dist1, dim=1) + torch.sum(dist2, dim=1)) / (torch.sum(valids, dim=1)+1e-7)
    loss_per_group_data = loss_per_group_data.reshape(batch_size, num_group)
    loss_per_group_data = loss_per_group_data * groups_valid
    loss_per_data = torch.sum(loss_per_group_data, dim=1) / torch.sum(groups_valid, dim=1)

    return loss_per_data


def get_total_box_loss(pts, quart_pred, quart_tgt, center_pred, center_tgt, valids, flat2group_index_1d, part_valids_group_mask, groups_valid, shape_box,
                      group_box, parts_box):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B, B x P
    """
    batch_size, num_part, num_point, _ = pts.size()
    batch_size, num_group, num_gpart, _ = part_valids_group_mask.size()

    flat2group_index_exp = flat2group_index_1d.unsqueeze(-1).repeat(1, 1, 3)

    center_pred = center_pred.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts_pred = qrot(quart_pred.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center_pred
    center_tgt = center_tgt.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts_tgt = qrot(quart_tgt.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center_tgt


    box_part_pred_min = pts_pred.view(-1, num_point, 3).min(1)[0]
    box_part_pred_max = pts_pred.view(-1, num_point, 3).max(1)[0]
    box_part_pred = torch.cat([(box_part_pred_min + box_part_pred_max)/2, (box_part_pred_max - box_part_pred_min).clamp(min=0.01)],dim=-1)
    box_part_tgt_min = pts_tgt.view(-1, num_point, 3).min(1)[0]
    box_part_tgt_max = pts_tgt.view(-1, num_point, 3).max(1)[0]
    box_part_tgt =  torch.cat([(box_part_tgt_min + box_part_tgt_max)/2, (box_part_tgt_max - box_part_tgt_min).clamp(min=0.01)], dim=-1)
    loss_box_parts_cal = get_loss_box(box_part_pred, box_part_tgt)
    loss_box_parts_cal = loss_box_parts_cal.view(batch_size, -1)*valids
    loss_box_per_data_cal = torch.sum(loss_box_parts_cal, dim=1) / torch.sum(valids, dim=1)


    loss_box_parts = get_loss_box(parts_box.reshape(batch_size*num_part, -1), box_part_tgt)
    loss_box_parts = loss_box_parts.view(batch_size, -1) * valids
    loss_box_per_data = torch.sum(loss_box_parts, dim=1) / torch.sum(valids, dim=1)



    gt_part_box_group_min = torch.gather(box_part_tgt_min.view(batch_size, num_part, 3), dim=1, index=flat2group_index_exp)
    gt_part_box_group_min = gt_part_box_group_min.view(batch_size, num_group, num_gpart, 3)
    gt_part_box_group_min = gt_part_box_group_min*part_valids_group_mask
    part_valids_group_mask_max = (part_valids_group_mask-1)*-10000
    gt_part_box_group_min = gt_part_box_group_min + part_valids_group_mask_max
    group_min = gt_part_box_group_min.min(2)[0]

    gt_part_box_group_max = torch.gather(box_part_tgt_max.view(batch_size, num_part, 3), dim=1, index=flat2group_index_exp)
    gt_part_box_group_max = gt_part_box_group_max.view(batch_size, num_group, num_gpart, 3)
    gt_part_box_group_max = gt_part_box_group_max*part_valids_group_mask
    part_valids_group_mask_min = (part_valids_group_mask-1)*10000
    gt_part_box_group_max = gt_part_box_group_max + part_valids_group_mask_min
    group_max = gt_part_box_group_max.max(2)[0]

    gt_part_box_group = torch.cat([(group_min + group_max) / 2, (group_max - group_min).clamp(min=0.01)], dim=-1)

    loss_box_group = get_loss_box(group_box.view(batch_size * num_group, -1), gt_part_box_group.view(batch_size * num_group, -1))
    loss_box_group = loss_box_group.view(batch_size, -1) * groups_valid
    loss_box_per_group = torch.sum(loss_box_group, dim=1) / torch.sum(groups_valid, dim=1)


    gt_shape_box_max = group_max.max(1)[0]
    gt_shape_box_min = group_min.min(1)[0]
    gt_shape_box = torch.cat([(gt_shape_box_max + gt_shape_box_min) / 2, (gt_shape_box_max - gt_shape_box_min).clamp(min=0.01)], dim=-1)
    loss_box_shape = get_loss_box(shape_box, gt_shape_box)


    loss_box = loss_box_per_data_cal + loss_box_per_data  + loss_box_per_group + loss_box_shape
    return loss_box


def get_loss_box(pred_boxes, target_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """


    loss_box_l = torch.abs(pred_boxes-target_boxes)
    loss_box_l = torch.mean(loss_box_l, dim = -1)






    loss_box_giou = 1 - (generalized_box_iou_match(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes)))



    loss_box = loss_box_l + loss_box_giou

    return loss_box