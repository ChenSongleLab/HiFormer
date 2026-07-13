
"""
    Scene Graph to predict the pose of each part
    adjust relation using the t in last iteration
    Input:
        relation matrxi of parts,part valids, part point clouds, instance label, iter_ind, pred_part_poses:      B x P x P, B x P, B x P x N x 3, B x P x P , (1 or 2 or 3) , B x P x 7
    Output:
        R and T:                B x P x (3 + 4)
    Losses:
        Center L2 Loss, Rotation L2 Loss, Rotation Chamder-Distance Loss
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet.pointnet_cls import get_model as pointnet_cls
from .pointnet.pointnet_cls import trans_loss
from .transformer import build_transformer_encoder, build_transformer_decoder
from .models import MLP, Predictor
from .losses import comp_losses, comp_decoder_losses_v2, comp_decoder_losses_freeze
from .inference import inference, decode_eval


class TransAssembly_freeze(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, args):
        super(TransAssembly_freeze, self).__init__()
        self.args = args

        if args.backbone == "pointnet_cls":
            self.feat_extract = pointnet_cls(k=args.feat_dim, normal_channel=False)
        else:
            raise NotImplementedError
        self.trans_criterion = trans_loss()

        self.encoder = build_transformer_encoder(args)
        if args.decode_on:
            self.decoder = build_transformer_decoder(args)

        assert self.args.encode_freeze
        assert self.args.decode_on







    def forward(self, part_pcs, part_valid, gt_part_poses, match_ids, part_ids,
                contact_points=None, sym_info=None):
        """
            Input:
                part_pcs: B x P x N x 3
                part_valid: B x P
                gt_part_poses: B x P x (3 + 4)
                relationships: B x P x P
                instance_labels:
                class_list:
            Output:
                pred_rt: B x P x (3 + 4) (T + R)
        """
        batch_size, num_part, _, _ = part_pcs.size()




        if self.args.encode_freeze:
            assert self.args.decode_on
            self.feat_extract.eval()
            self.encoder.eval()
            with torch.no_grad():
                base_feat, trans_feat = self.feat_extract(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
        else:
            base_feat, trans_feat = self.feat_extract(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
            trans_loss = self.trans_criterion(trans_feat)

        base_feat = base_feat.view(batch_size, num_part, -1)
        if self.training:

            if not self.args.encode_freeze:
                if self.args.filter_on:
                    prob = torch.rand(1).item()
                    if prob < self.args.filter_thresh:
                        base_feat, part_pcs, part_valid, gt_part_poses, part_ids, match_ids, _, _, part_mask = \
                            self.prepare_filters_v1(base_feat, part_pcs, part_valid, gt_part_poses, part_ids, match_ids)


            if self.args.decode_on:


                decode_feat, decode_pcs, decode_mask, positive_ids = \
                    self.prepare_decoder_v2(base_feat.clone(), part_pcs.clone(), part_valid)

            output = dict()
            kwargs = {"part_ids": part_ids}
            for mon_id in range(self.args.train_mon):

                if self.args.encode_freeze:
                    with torch.no_grad():
                        preds, memory = self.encoder(base_feat, part_valid, None, None, **kwargs)
                else:
                    preds, memory = self.encoder(base_feat, part_valid, None, None, **kwargs)


                if self.args.decode_on:
                    if self.args.feat_in_detach:
                        with torch.no_grad():
                            decode_feat = decode_feat.clone()
                    if self.args.memory_detach:
                        with torch.no_grad():
                            memory = memory.detach()
                    memory, memory_poses, decode_poses, decode_cates = self.prepare_labels_v1(memory, preds, gt_part_poses, decode_mask, positive_ids)

                    kwargs["decode_mask"] = decode_mask
                    kwargs["decode_ids"] = positive_ids
                    kwargs["memory_poses"] = memory_poses
                    pred_poses, _ = self.decoder(decode_feat, memory, part_valid, None, None, **kwargs)
                    fg_preds = self.prepare_fg_pred(pred_poses, preds, positive_ids)
                    loss_per_mon, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss = \
                        comp_decoder_losses_freeze(fg_preds, gt_part_poses, part_pcs, part_valid, match_ids, decode_mask, args=self.args)


                if mon_id == 0:
                    loss = loss_per_mon.clone()
                else:
                    loss = torch.min(loss, loss_per_mon)

            if self.args.encode_freeze:
                trans_loss = loss.new_zeros(1)[0]
            loss += trans_loss
            return preds, loss, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss, trans_loss, output

        else:
            output = dict()
            if self.args.type_eval == "decoder":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_decoder(base_feat, part_pcs, part_valid, gt_part_poses,
                                           part_ids, match_ids, contact_points, sym_info)
            elif self.args.type_eval == "wip":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_wip(base_feat, part_pcs, part_valid, gt_part_poses,
                                       part_ids, match_ids, contact_points, sym_info)
            elif self.args.type_eval == "encoder":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_encoder(base_feat, part_pcs, part_valid, gt_part_poses,
                                           part_ids, match_ids, contact_points, sym_info)
            else:
                raise NotImplementedError

            return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, num_contact_correct, num_contact_point, batch_size, output

    def prepare_decoder_v1(self, part_feat, part_pcs, part_valid):
        """
        Positive + Negative.
        Positive is from self.
        Negative is from random other in a batch.
        Support multi positive (>1), but no semi-positive(intra competition).
        """
        batch_size, num_part, num_channel = part_feat.size()
        _, _, num_point, _ = part_pcs.size()

        if self.args.rand_pos:
            num_pos = torch.randperm(self.args.num_pos)[0].item() + 1
        else:
            num_pos = self.args.num_pos


        rand_ids = [torch.randperm(int(bs)) for bs in part_valid.sum(1)]
        pos_ids = torch.cat([ids[:num_pos] for ids in rand_ids]).to(part_valid.device)
        pos_bs = torch.tensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, num_pos).view(-1).to(part_valid.device)
        neg_ids = torch.cat([ids[num_pos:] for ids in rand_ids]).to(part_valid.device)
        neg_bs_ = [[bs_id] * int(bs_num - num_pos) for bs_id, bs_num in enumerate(part_valid.sum(1))]
        neg_bs = []
        for bs_id in range(batch_size):
            neg_bs += neg_bs_[bs_id]
        neg_bs = torch.tensor(neg_bs).to(part_valid.device)


        decode_feat = part_feat.new_zeros((batch_size, self.args.num_queries, num_channel))
        decode_pcs = part_pcs.new_zeros((batch_size, self.args.num_queries, num_point, 3))

        pos_feat = part_feat[pos_bs, pos_ids].view(batch_size, num_pos, -1)
        decode_feat[:, :num_pos, :] = pos_feat
        pos_pcs = part_pcs[pos_bs, pos_ids].view(batch_size, num_pos, num_point, -1)
        decode_pcs[:, :num_pos, :, :] = pos_pcs

        neg_feat_gallery = part_feat[neg_bs, neg_ids]
        neg_pcs_gallery = part_pcs[neg_bs, neg_ids]
        num_neg = self.args.num_queries - num_pos
        for bs_id in range(batch_size):
            neg_mask = neg_bs != bs_id
            neg_feat = neg_feat_gallery[neg_mask]
            neg_pcs = neg_pcs_gallery[neg_mask]
            total_neg = neg_feat.size(0)
            if total_neg >= num_neg:
                select_ids = torch.randperm(total_neg)[:num_neg].to(part_valid.device)
                decode_feat[bs_id, num_pos:, :] = neg_feat[select_ids, :]
                decode_pcs[bs_id, num_pos:, :, :] = neg_pcs[select_ids, :, :]
            else:
                decode_feat[bs_id, -total_neg:, :] = neg_feat.clone()
                decode_pcs[bs_id, -total_neg:, :, :] = neg_pcs.clone()


        decode_mask = part_feat.new_ones((batch_size, num_part, 1))
        decode_mask[pos_bs, pos_ids] = 0.
        decode_mask = decode_mask.transpose(0, 1).bool()

        return decode_feat, decode_pcs, decode_mask, pos_ids.view(batch_size, num_pos)

    def prepare_decoder_v2(self, part_feat, part_pcs, part_valid):
        """
        Positive only.
        Support multi positive (>1), but no semi-positive(intra competition).
        """
        batch_size, num_part, num_channel = part_feat.size()
        _, _, num_point, _ = part_pcs.size()

        if self.args.rand_pos:
            num_pos = torch.randperm(self.args.num_pos)[0].item() + 1
        else:
            num_pos = self.args.num_pos


        if self.training:
            rand_ids = [torch.randperm(int(bs)) for bs in part_valid.sum(1)]
            pos_ids = torch.cat([ids[:num_pos] for ids in rand_ids]).to(part_valid.device)
        else:
            raise NotImplementedError
        pos_bs = torch.tensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, num_pos).view(-1).to(part_valid.device)


        decode_feat = part_feat[pos_bs, pos_ids].view(batch_size, num_pos, -1).contiguous()
        decode_pcs = part_pcs[pos_bs, pos_ids].view(batch_size, num_pos, num_point, -1).contiguous()


        decode_mask = part_feat.new_ones((batch_size, num_part, 1))
        decode_mask[pos_bs, pos_ids] = 0.
        decode_mask = decode_mask.transpose(0, 1).bool()

        return decode_feat, decode_pcs, decode_mask, pos_ids.view(batch_size, num_pos)

    def prepare_labels_v1(self, memory, pred_poses, gt_poses, decode_mask, pos_ids):
        """
        """
        num_part, batch_size, len_feat = memory.size()
        _, num_pos = pos_ids.size()


        decode_memory = memory[decode_mask.repeat(1, 1, len_feat)].view(num_part - num_pos, batch_size, len_feat)


        memory_poses = pred_poses[:, -1].transpose(0, 1).contiguous()
        memory_poses = memory_poses[decode_mask.repeat(1, 1, 7)].view(num_part - num_pos, batch_size, 7).detach()


        pos_ids = pos_ids.view(-1)
        pos_bs = torch.arange(batch_size).unsqueeze(1).repeat(1, num_pos).view(-1).to(gt_poses.device)
        decode_poses = gt_poses[pos_bs, pos_ids, :].view(batch_size, num_pos, 7)


        cate_labels = gt_poses.new_zeros((batch_size, self.args.num_queries, 1))
        cate_labels[:, :num_pos, :] = 1.

        return decode_memory, memory_poses, decode_poses, cate_labels

    def prepare_fg_pred(self, decode_pred, encode_pred, pos_ids):
        batch_size, num_trans, num_part, len_feat = encode_pred.size()
        _, num_pos = pos_ids.size()

        pos_ids = pos_ids.view(-1)
        pos_bs = torch.arange(batch_size).unsqueeze(1).repeat(1, num_pos).view(-1).to(pos_ids.device)

        fg_pred = encode_pred.clone()
        fg_pred[pos_bs, :, pos_ids, :] = decode_pred.squeeze(2)

        return fg_pred

    def generate_fg_pred(self, part_pred, wip_pred, pos_id):
        batch_size, num_trans, num_part_wip, len_feat = wip_pred.size()
        pos_ids = part_pred.new_ones((batch_size, 1)).long() * pos_id

        pos_ids = pos_ids.view(-1)
        pos_bs = torch.arange(batch_size).to(pos_ids.device)

        num_part = num_part_wip + 1
        fg_pred = part_pred.new_zeros((batch_size, num_trans, num_part, len_feat))
        if pos_id == 0:
            fg_pred[:, :, 1:, :] = wip_pred
        elif pos_id == num_part - 1:
            fg_pred[:, :, :-1, :] = wip_pred
        else:
            fg_pred[:, :, :pos_id, :] = wip_pred[:, :, :pos_id, :]
            fg_pred[:, :, pos_id+1:, :] = wip_pred[:, :, pos_id:, :]
        fg_pred[pos_bs, :, pos_ids, :] = part_pred.squeeze(2)

        return fg_pred

    def prepare_filters_v1(self, part_feat, part_pcs, part_valid, part_pose, part_ids, match_ids,
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

        kwargs = {"part_ids": part_ids}
        for mon_id in range(self.args.eval_mon):

            preds, _ = self.encoder(base_feat, part_valid, None, None, **kwargs)


            part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                acc_per_mon, contact_correct_per_mon, num_contact_point = \
                inference(preds, part_poses, part_pcs, part_valid, match_ids,
                          contact_points, sym_info, args=self.args)


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

        part_cd_loss = part_cd_loss.mean()
        shape_cd_loss = shape_cd_loss.mean()
        contact_point_loss = contact_point_loss.mean()
        acc = acc.sum()
        valid = part_valid.sum()
        num_contact_correct = num_contact_correct.sum()
        num_contact_point = num_contact_point.sum()
        return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
               num_contact_correct, num_contact_point, batch_size

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
                self.prepare_filters_v1(base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids,
                                        contact_points, sym_info, filter_id)
            kwargs = {"part_ids": cur_part_ids}
            for mon_id in range(self.args.eval_mon):

                preds, _ = self.encoder(cur_base_feat, cur_part_valid, None, None, **kwargs)


                part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
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

    def inference_decoder(self, base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids, contact_points, sym_info):
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
                   self.prepare_filters_v1(base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids,
                                           contact_points, sym_info, filter_id)
            for mon_id in range(self.args.eval_mon):

                kwargs = {"part_ids": cur_part_ids}
                preds, memory = self.encoder(cur_base_feat, cur_part_valid, None, None, **kwargs)


                decode_feat = base_feat[(~part_mask).unsqueeze(-1).repeat(1, 1, len_feat)].view(batch_size, 1, len_feat).contiguous()
                kwargs = {"part_ids": part_ids}
                decode_ids = preds.new_ones((batch_size, 1)).long() * filter_id
                decode_bs = torch.arange(batch_size).to(decode_ids.device)
                kwargs["decode_ids"] = decode_ids
                decode_mask = preds.new_ones((batch_size, num_part, 1))
                decode_mask[decode_bs, decode_ids] = 0.
                decode_mask = decode_mask.transpose(0, 1).bool()
                kwargs["decode_mask"] = decode_mask
                memory_poses = preds[:, -1].transpose(0, 1).contiguous()
                kwargs["memory_poses"] = memory_poses
                pred_poses, _ = self.decoder(decode_feat, memory, part_valid, None, None, **kwargs)


                fg_preds = self.generate_fg_pred(pred_poses, preds, filter_id)


                part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                    acc_per_mon, contact_correct_per_mon, num_contact_point = \
                    decode_eval(fg_preds, part_poses, part_pcs, part_valid, match_ids,
                                contact_points, sym_info, part_mask, args=self.args)


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
            valid_wip += part_valid[~part_mask] * ins_valid
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
