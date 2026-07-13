
"""
    Scene Graph framework to predict the pose of each part
    Input:
    Output:
        R and T:  B x P x (3 + 4)
    Losses:
        Center L2 Loss, Rotation L2 Loss, Rotation Chamder-Distance Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet.pointnet_cls import get_model as pointnet_cls
from .pointnet.pointnet_cls import trans_loss

from .gnn import AssemblyGNN
from .models import MLP, Predictor
from .losses import comp_losses
from .inference import inference


class TransAssembly_gnn(nn.Module):
    """
    """
    def __init__(self, args):
        super(TransAssembly_gnn, self).__init__()
        self.args = args

        if args.backbone == "pointnet_cls":
            self.feat_extract = pointnet_cls(k=args.feat_dim, normal_channel=False)
        else:
            raise NotImplementedError
        self.trans_criterion = trans_loss()

        self.gnn = AssemblyGNN(args)

    def forward(self, part_pcs, part_valid, gt_part_poses, match_ids, part_ids,
                contact_points=None, sym_info=None):
        """
            Input:
                part_pcs: B x P x N x 3
                part_valid: B x P
                gt_part_poses: B x P x (3 + 4)

            Output:
                pred_rt: B x P x (3 + 4) (T + R)
        """



        batch_size, num_part, _, _ = part_pcs.size()
        base_feat, trans_feat = self.feat_extract(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
        base_feat = base_feat.view(batch_size, num_part, -1)
        trans_loss = self.trans_criterion(trans_feat)

        if self.training:

            if self.args.filter_on:
                flag_valid = (part_valid.sum(1) == 1).sum().bool()
                if flag_valid:
                    pass
                else:
                    prob = torch.rand(1).item()
                    if prob < self.args.filter_thresh:
                        base_feat, part_pcs, part_valid, gt_part_poses, part_ids, match_ids, _, _, part_mask = \
                            self.prepare_filters(base_feat, part_pcs, part_valid, gt_part_poses, part_ids, match_ids)

            output = dict()
            kwargs = {"part_ids": part_ids}
            for mon_idx in range(self.args.train_mon):

                preds = self.gnn(base_feat, part_valid, **kwargs)
                loss_per_mon, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss = \
                    comp_losses(preds, gt_part_poses, part_pcs, part_valid, match_ids, args=self.args)
                if mon_idx == 0:
                    loss = loss_per_mon.clone()
                else:
                    loss = torch.min(loss, loss_per_mon)
            loss += trans_loss
            return preds, loss, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss, trans_loss, output

        else:
            output = dict()
            if self.args.type_eval == "encoder":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size, pred_poses = \
                    self.inference_encoder(base_feat, part_pcs, part_valid, gt_part_poses,
                                           part_ids, match_ids, contact_points, sym_info)
                output["pred_poses"] = pred_poses
            elif self.args.type_eval == "wip":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_wip(base_feat, part_pcs, part_valid, gt_part_poses,
                                       part_ids, match_ids, contact_points, sym_info)
            else:
                raise NotImplementedError

            return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, num_contact_correct, num_contact_point, batch_size, output

    def prepare_filters(self, part_feat, part_pcs, part_valid, part_pose, part_ids, match_ids,
                           contact_points=None, sym_info=None, filter_id=0):
        """
        Now only support num_filter == n.
        """
        batch_size, num_part, len_feat = part_feat.size()
        _, _, num_point, _ = part_pcs.size()
        num_filter = self.args.num_filter
        num_res = num_part - num_filter

        part_mask = part_feat.new_ones((batch_size, num_part))
        if self.training:
            rand_ids = [torch.randperm(int(bs)) for bs in part_valid.sum(1)]
            filter_ids = torch.cat([ids[:num_filter] for ids in rand_ids]).to(part_valid.device)
        else:
            filter_ids = torch.tensor([filter_id for _ in range(batch_size)]).to(part_valid.device)
        filter_bs = torch.tensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, num_filter).view(-1).to(part_valid.device)
        part_mask[filter_bs, filter_ids] = 0.
        part_mask = part_mask.bool()
        part_mask_2 = part_mask.clone()
        part_mask_3 = part_mask.unsqueeze(-1)
        part_mask_4 = part_mask.unsqueeze(-1).unsqueeze(-1)

        part_feat = part_feat[part_mask_3.repeat(1, 1, len_feat)].view(batch_size, num_res, len_feat).contiguous()
        part_pcs = part_pcs[part_mask_4.repeat(1, 1, num_point, 3)].view(batch_size, num_res, num_point, 3).contiguous()
        part_valid = part_valid[part_mask_2].view(batch_size, num_res).contiguous()
        part_pose = part_pose[part_mask_3.repeat(1, 1, 7)].view(batch_size, num_res, 7).contiguous()
        part_ids = part_ids[part_mask_2].view(batch_size, num_res).contiguous()
        if sym_info is not None:
            sym_info = sym_info[part_mask_3.repeat(1, 1, 3)].view(batch_size, num_res, 3).contiguous()


        if contact_points is not None:
            contact_points = contact_points[part_mask_4.repeat(1, 1, num_part, 4)].view(batch_size, num_res, num_part, 4).contiguous()
            part_mask_4_ = part_mask.unsqueeze(1).unsqueeze(-1).repeat(1, num_res, 1, 4)
            contact_points = contact_points[part_mask_4_].view(batch_size, num_res, num_res, 4).contiguous()


        filter_match_ids = list()
        part_mask = part_mask.cpu().numpy()
        for bs, match_id in enumerate(match_ids):
            filter_match_ids.append(match_id[part_mask[bs]])

        return part_feat, part_pcs, part_valid, part_pose, part_ids, filter_match_ids, contact_points, sym_info, part_mask_2

    def inference_encoder(self, base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids, contact_points, sym_info):
        batch_size, num_part, len_feat = base_feat.size()

        pred_poses = list()
        measures = list()
        kwargs = {"part_ids": part_ids}
        for mon_id in range(self.args.eval_mon):

            preds = self.gnn(base_feat, part_valid, **kwargs)


            pred_poses_per_mon, part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                acc_per_mon, contact_correct_per_mon, num_contact_point = \
                inference(preds, part_poses, part_pcs, part_valid, match_ids,
                          contact_points, sym_info, args=self.args)


            if mon_id == 0:
                part_cd_loss = part_cd_loss_per_mon.clone()
                shape_cd_loss = shape_cd_loss_per_mon.clone()
                contact_point_loss = contact_point_loss_per_mon.clone()
                acc = acc_per_mon.clone()
                num_contact_correct = contact_correct_per_mon.clone()
            elif self.args.worst_mon:
                part_cd_loss = part_cd_loss.max(part_cd_loss_per_mon)
                shape_cd_loss = shape_cd_loss.max(shape_cd_loss_per_mon)
                contact_point_loss = contact_point_loss.max(contact_point_loss_per_mon)
                acc = acc.min(acc_per_mon)
                num_contact_correct = num_contact_correct.min(contact_correct_per_mon)
            else:

                part_cd_loss = part_cd_loss.min(part_cd_loss_per_mon)
                shape_cd_loss = shape_cd_loss.min(shape_cd_loss_per_mon)
                contact_point_loss = contact_point_loss.min(contact_point_loss_per_mon)
                acc = acc.max(acc_per_mon)
                num_contact_correct = num_contact_correct.max(contact_correct_per_mon)

            pred_poses.append(pred_poses_per_mon.unsqueeze(1))
            measures.append(acc_per_mon.unsqueeze(1))

        part_cd_loss = part_cd_loss.mean()
        shape_cd_loss = shape_cd_loss.mean()
        contact_point_loss = contact_point_loss.mean()
        acc = acc.sum()
        valid = part_valid.sum()
        num_contact_correct = num_contact_correct.sum()
        num_contact_point = num_contact_point.sum()


        pred_poses = torch.cat(pred_poses, dim=1)
        measures = torch.cat(measures, dim=1)
        if self.args.pred_encoder_vis:
            _, sort_indices = measures.sort(dim=1, descending=True)
            for bs_id in range(batch_size):
                pred_poses[bs_id] = pred_poses[bs_id, sort_indices[bs_id]]

        return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
               num_contact_correct, num_contact_point, batch_size, pred_poses

    def inference_wip(self, base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids, contact_points, sym_info):
        batch_size, num_part, len_feat = base_feat.size()


        part_cd_loss_wip = base_feat.new_zeros(batch_size)
        shape_cd_loss_wip = base_feat.new_zeros(batch_size)
        contact_point_loss_wip = base_feat.new_zeros(batch_size)
        acc_wip = base_feat.new_zeros(batch_size)
        valid_wip = base_feat.new_zeros(batch_size)
        num_contact_correct_wip = base_feat.new_zeros(batch_size)
        num_contact_point_wip = base_feat.new_zeros(batch_size)
        ins_valid_wip = base_feat.new_zeros(batch_size)


        num_part_per_ins = part_valid.sum(1)
        num_part_max = num_part_per_ins.max().long().item()
        for filter_id in range(num_part_max):
            cur_base_feat, cur_part_pcs, cur_part_valid, cur_part_poses, \
            cur_part_ids, cur_match_ids, cur_contact_points, cur_sym_info, part_mask = \
                self.prepare_filters(base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids,
                                        contact_points, sym_info, filter_id)
            kwargs = {"part_ids": cur_part_ids}
            for mon_id in range(self.args.eval_mon):

                preds, _ = self.encoder(cur_base_feat, cur_part_valid, None, None, **kwargs)


                _, part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                acc_per_mon, contact_correct_per_mon, num_contact_point = \
                    inference(preds, cur_part_poses, cur_part_pcs, cur_part_valid, cur_match_ids,
                              cur_contact_points, cur_sym_info, args=self.args)


                if mon_id == 0:
                    part_cd_loss = part_cd_loss_per_mon.clone()
                    shape_cd_loss = shape_cd_loss_per_mon.clone()
                    contact_point_loss = contact_point_loss_per_mon.clone()
                    acc = acc_per_mon.clone()
                    num_contact_correct = contact_correct_per_mon.clone()
                else:

                    part_cd_loss = part_cd_loss.min(part_cd_loss_per_mon)
                    shape_cd_loss = shape_cd_loss.min(shape_cd_loss_per_mon)
                    contact_point_loss = contact_point_loss.min(contact_point_loss_per_mon)
                    acc = acc.max(acc_per_mon)
                    num_contact_correct = num_contact_correct.max(contact_correct_per_mon)

            ins_valid = (num_part_per_ins > filter_id).float()
            part_cd_loss_wip += part_cd_loss * ins_valid
            shape_cd_loss_wip += shape_cd_loss * ins_valid
            contact_point_loss_wip += contact_point_loss * ins_valid
            acc_wip += acc * ins_valid
            valid_wip += cur_part_valid.sum(-1) * ins_valid
            num_contact_correct_wip += num_contact_correct * ins_valid
            num_contact_point_wip += num_contact_point * ins_valid
            ins_valid_wip += ins_valid

        part_cd_loss = (part_cd_loss_wip / ins_valid_wip).mean()
        shape_cd_loss = (shape_cd_loss_wip / ins_valid_wip).mean()
        contact_point_loss = (contact_point_loss_wip / ins_valid_wip).mean()
        acc = (acc_wip / ins_valid_wip).sum()
        valid = (valid_wip / ins_valid_wip).sum()
        num_contact_correct = (num_contact_correct_wip / ins_valid_wip).sum()
        num_contact_point = (num_contact_point_wip / ins_valid_wip).sum()
        return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
               num_contact_correct, num_contact_point, batch_size
