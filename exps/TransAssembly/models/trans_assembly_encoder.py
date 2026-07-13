
"""
    Transformer-encoder-based framework to predict the pose of each part
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
from .transformer import build_transformer_encoder, build_transformer_decoder
from .models import MLP, Predictor
from .losses import comp_losses, comp_decoder_losses, comp_group_losses, comp_group_losses_fast
from .inference import inference
from model.swin3d_transformer import Swin
import torch_points_kernels as tp
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
from quaternion import qeuler,qrot
from .box_ops import *







def show_process_gpu_info(prefix = "", show = True):
    current_gpu_index = 0
    total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)
    process_used_tensor_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
    process_used_total_memory = torch.cuda.memory_reserved(current_gpu_index) / (1024 ** 3)
    free_memory = total_memory - process_used_total_memory
    if show:
        print(
        f"{prefix:s}-GPU mem total：{total_memory:.2f} GB, GPU mem free：{free_memory:.2f} GB, tensor used：{process_used_tensor_memory:.2f} GB, process used：{process_used_total_memory:.2f} GB")

    return free_memory




class TransAssembly_encoder(nn.Module):
    """
    """
    def __init__(self, args):
        super(TransAssembly_encoder, self).__init__()

        args.backbone = args.backbone.split(",")
        self.args = args
        backbones = len(args.backbone)
        if backbones > 1:
            self.linear1 = nn.Linear(args.feat_dim*backbones, 2048)
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(2048, args.feat_dim)
            self.activation = F.relu

        if  "pointnet_cls" in args.backbone:
            self.feat_extract = pointnet_cls(k=args.feat_dim, normal_channel=False)
            self.trans_criterion = trans_loss()
        if "swin_transformer" in args.backbone:

            args.patch_size = args.grid_size * args.patch_size
            args.window_sizes = [args.patch_size * args.window_size * (2 ** i) for i in range(args.num_layers)]
            args.grid_sizes = [args.patch_size * (2 ** i) for i in range(args.num_layers)]
            args.quant_sizes = [args.quant_size * (2 ** i) for i in range(args.num_layers)]
            self.swin_tm = Swin(args.depths, args.channels, args.num_heads, \
                         args.window_sizes, args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
                         rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, \
                         concat_xyz=args.concat_xyz, num_classes=args.classes, \
                         ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)




        self.encoder = build_transformer_encoder(args)
        if args.decode_on:
            self.decoder = build_transformer_decoder(args)


    def feature_extract_swin_transformer(self, part_pcs, parts_valid):
        time_start = time.time()
        bs, num_part, num_point, num_pos = part_pcs.size()
        offset, count = [], 0
        part_pcs_valid = []
        dense2flat_index = np.zeros((bs, num_part), dtype=np.int64)
        start_pos = 0
        for i in range(bs):
            for j in range(int(parts_valid[i].sum())):
                count += num_point
                offset.append(count)
            part_pcs_valid.append(part_pcs[i][:int(parts_valid[i].sum())])
            dense2flat_index[i, :int(parts_valid[i].sum())] = range(start_pos, start_pos+int(parts_valid[i].sum()))
            start_pos += int(parts_valid[i].sum())
        part_pcs_valid = torch.cat(part_pcs_valid)
        part_pcs_valid = part_pcs_valid.view(-1, num_pos)
        coord, feat = part_pcs_valid + 0.5, part_pcs_valid


        offset = torch.IntTensor(offset).cuda()
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long().cuda()


        time_start = time.time()
        args = self.args
        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]


        time_start = time.time()
        coord, feat, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        output,loss_points = self.swin_tm(feat, coord, offset, batch, neighbor_idx)
        dense2flat_index = torch.from_numpy(dense2flat_index).view(bs*num_part).unsqueeze(-1).repeat(1, 256).cuda(non_blocking=True)
        out_feat = torch.gather(output, dim=0, index=dense2flat_index)


        time_start = time.time()
        return out_feat, dense2flat_index, loss_points


    def forward(self, shape_id, part_pcs, part_boxes, part_valid, gt_part_poses, match_ids, part_ids, num_group, num_group_part, parts_valid_in_group, \
                groups_valid, flat2group_index, group2flat_index, contact_points=None, sym_info=None):
        """
            Input:
                part_pcs: B x P x N x 3
                part_valid: B x P
                gt_part_poses: B x P x (3 + 4)
            Output:
                pred_rt: B x P x (3 + 4) (T + R)
        """



        trans_loss = None
        batch_size, num_part, _, _ = part_pcs.size()
        base_feat_list = []
        if  "pointnet_cls" in self.args.backbone:
            base_feat_point, trans_feat = self.feat_extract(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
            trans_loss = self.trans_criterion(trans_feat)
            base_feat_point = base_feat_point.view(batch_size, num_part, -1)
            base_feat_list.append(base_feat_point)
        if  "swin_transformer" in self.args.backbone:
            base_feat_swing , dense2flat_index, loss_swin_points = self.feature_extract_swin_transformer(part_pcs, part_valid)
            base_feat_swing = base_feat_swing.view(batch_size, num_part, -1)
            base_feat_list.append(base_feat_swing)
            trans_loss = loss_swin_points
        if len(base_feat_list) > 1:
            base_feat_cat = torch.cat(base_feat_list, dim = 2)
            base_feat = self.linear2(self.dropout(self.activation(self.linear1(base_feat_cat))))
        else:
            base_feat = base_feat_list[0]
        dense2flat_index = self.get_dense2flat_index(batch_size, num_part, part_valid)
        base_feat = base_feat.view(batch_size, num_part, -1)


        kwargs = {"part_ids": part_ids, "part_boxes": part_boxes, "num_group": num_group, "num_group_part": num_group_part, "parts_valid_in_group": parts_valid_in_group, \
                  "groups_valid": groups_valid, "flat2group_index": flat2group_index, "group2flat_index": group2flat_index, "dense2flat_index":dense2flat_index}

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



            for mon_idx in range(self.args.train_mon):

                preds, _ = self.encoder(base_feat, part_valid, None, None, **kwargs)
                loss_per_mon, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss, box_loss, group_cd_loss, box_vertex_loss = \
                    comp_group_losses_fast(preds, gt_part_poses, part_pcs, part_valid, match_ids, args=self.args, **kwargs)
                if mon_idx == 0:
                    loss = loss_per_mon.clone()
                else:
                    loss = torch.min(loss, loss_per_mon)
            if trans_loss != None:
                loss += trans_loss
            else:
                trans_loss = torch.zeros_like(shape_cd_loss)
            return preds, loss, trans_l2_loss, rot_l2_loss, rot_cd_loss, shape_cd_loss, trans_loss, box_loss, group_cd_loss, output

        else:
            if shape_id[0] == 2514:
                print("hello")
            output = dict()
            if self.args.type_eval == "encoder":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size, pred_poses, shapecd_dist1, shapecd_dist2 = \
                    self.inference_encoder(base_feat, part_pcs, part_valid, gt_part_poses,
                                            match_ids, contact_points, sym_info, **kwargs)
                output["pred_poses"] = pred_poses
            elif self.args.type_eval == "wip":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_wip(base_feat, part_pcs, part_valid, gt_part_poses,
                                       part_ids, match_ids, contact_points, sym_info)
            else:
                raise NotImplementedError
            if False:
                self.show_results(shape_id, part_valid, part_pcs, part_boxes, pred_poses[:, -1],  gt_part_poses, shape_cd_loss, part_ids, shapecd_dist1, shapecd_dist2)
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

    def inference_encoder(self, base_feat, part_pcs, part_valid, part_poses, match_ids, contact_points, sym_info, **kwargs):
        batch_size, num_part, len_feat = base_feat.size()

        pred_poses = list()
        measures = list()

        part_ids = kwargs["part_ids"]
        for mon_id in range(self.args.eval_mon):

            preds, _ = self.encoder(base_feat, part_valid, None, None, **kwargs)


            pred_poses_per_mon, part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                acc_per_mon, contact_correct_per_mon, num_contact_point, shapecd_dist1, shapecd_dist2 = \
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
               num_contact_correct, num_contact_point, batch_size, pred_poses, shapecd_dist1, shapecd_dist2

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

    def get_dense2flat_index(self, bs, num_part, parts_valid):
        dense2flat_index = np.zeros((bs, num_part), dtype=np.int64)
        start_pos = 0
        for i in range(bs):
            dense2flat_index[i, :int(parts_valid[i].sum())] = range(start_pos, start_pos + int(parts_valid[i].sum()))
            start_pos += int(parts_valid[i].sum())
        dense2flat_index = torch.from_numpy(dense2flat_index).view(bs*num_part).unsqueeze(-1).repeat(1, 256).cuda(non_blocking=True)
        return dense2flat_index





    def render_parts_pcs(self, pts, rows = 1, cols = 1, index = 1, xyzrange = None, fig = None, show = True, save_path=None, part_ids=None):






        pts = pts.cpu().detach().numpy()
        part_ids = part_ids.cpu().detach().numpy()

        if fig == None:
            ax = plt.figure().add_subplot(rows, cols, index, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, index, projection='3d')
        if list(xyzrange) != None:
            ax.set_xlim(xyzrange[0], xyzrange[1])
            ax.set_ylim(xyzrange[2], xyzrange[3])
            ax.set_zlim(xyzrange[4], xyzrange[5])
        for i in range(pts.shape[0]):
            ax.scatter(pts[i][:,0], pts[i][:,1], pts[i][:,2], marker='o', c=plt.cm.Set1(part_ids[i]/10.0))
        if show:
            plt.show()
        if save_path != None:
            plt.savefig(save_path)

    def render_parts_pcs_pred_gt(self, pts, pts_gt,rows = 1, cols = 1, index = 1, xyzrange = None, fig = None, show = True, save_path=None, part_ids=None):






        pts = pts.cpu().detach().numpy()
        pts_gt = pts_gt.cpu().detach().numpy()
        part_ids = part_ids.cpu().detach().numpy()

        if fig == None:
            ax = plt.figure().add_subplot(rows, cols, index, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, index, projection='3d')
        if list(xyzrange) != None:
            ax.set_xlim(xyzrange[0], xyzrange[1])
            ax.set_ylim(xyzrange[2], xyzrange[3])
            ax.set_zlim(xyzrange[4], xyzrange[5])
        for i in range(pts.shape[0]):
            ax.scatter(pts[i][:,0], pts[i][:,1], pts[i][:,2], marker='o',c=plt.cm.Set1(part_ids[i]/10.0))
            ax.scatter(pts_gt[i][:,0], pts_gt[i][:,1], pts_gt[i][:,2], marker='o',c=plt.cm.Set1(part_ids[i]/10.0))

        if show:
            plt.show()
        if save_path != None:
            plt.savefig(save_path)

    def render_parts_pcs_pred_gt_in_one_image(self, pts, pts_gt, box, box_gt, rows = 1, cols = 1, index = 1, xyzrange = None, fig = None, show = False, save_path=None, part_ids=None, shape_cd_loss=None, shapecd_dist1=None, shapecd_dist2=None):







        dist = shapecd_dist1+shapecd_dist2
        dist = dist.view(pts.shape[0], -1)
        dist = dist.cpu().detach().numpy()
        shape_cd_loss = shape_cd_loss.cpu().detach().numpy()
        show_index = np.where(dist > shape_cd_loss, True, False)
        pts = pts.cpu().detach().numpy()
        pts_gt = pts_gt.cpu().detach().numpy()
        part_ids = part_ids.cpu().detach().numpy()


        rows = 2
        cols = 2
        index = 1

        fig = plt.figure()
        for index in range(1,rows*cols+1):
            ax = fig.add_subplot(rows, cols, index, projection='3d')
            if list(xyzrange) != None:
                ax.set_xlim(xyzrange[0], xyzrange[1])
                ax.set_ylim(xyzrange[2], xyzrange[3])
                ax.set_zlim(xyzrange[4], xyzrange[5])

            for i in range(pts.shape[0]):
                if index == 1:
                    ax.scatter(pts[i][:,0], pts[i][:,1], pts[i][:,2], marker='o',c=plt.cm.Set1(part_ids[i]/10.0))
                    for l in box:
                        ax.plot(l[0], l[1], l[2], c=plt.cm.Set1(part_ids[i]/10.0))
                if index == 2:
                    ax.scatter(pts_gt[i][:,0], pts_gt[i][:,1], pts_gt[i][:,2], marker='o',c=plt.cm.Set1(part_ids[i]/10.0))
                    for l in box_gt:
                        ax.plot(l[0], l[1], l[2], c=plt.cm.Set1(part_ids[i]/10.0))
                if index == 3:
                    ax.scatter(pts[i][:, 0], pts[i][:, 1], pts[i][:, 2], marker='o', c=plt.cm.Set1(part_ids[i] / 10.0))
                    ax.scatter(pts_gt[i][:, 0], pts_gt[i][:, 1], pts_gt[i][:, 2], marker='o', c=plt.cm.Set1(part_ids[i] / 10.0))
                if index == 4:
                    ax.scatter(pts[i][:, 0], pts[i][:, 1], pts[i][:, 2], marker='o', c='r')
                    ax.scatter(pts_gt[i][:, 0], pts_gt[i][:, 1], pts_gt[i][:, 2], marker='o', c='g')
                    ax.scatter(pts[i,show_index[i], 0], pts[i,show_index[i], 1], pts[i,show_index[i], 2], marker='o', c='b')
                    ax.scatter(pts_gt[i, show_index[i], 0], pts_gt[i, show_index[i], 1], pts_gt[i, show_index[i], 2], marker='o', c='m')

        if show:
            plt.show()
        if save_path != None:
            plt.savefig(save_path)
        plt.close(fig)

    def show_results(self, shape_id,  part_valid, part_pcs, part_boxes, pred_poses, gt_poses, shape_cd_loss,part_ids, shapecd_dist1, shapecd_dist2):

        box_vertex = box_xyxy_to_vextex(box_cxcywh_to_xyxy(part_boxes))

        center = pred_poses[:, :, :3]
        quat = pred_poses[:, :, 3:]
        center_rep = center.unsqueeze(2).repeat(1, 1, part_pcs.shape[2], 1)
        quat_rep = quat.unsqueeze(2).repeat(1, 1, part_pcs.shape[2], 1)
        part_pcs_pred = qrot(quat_rep, part_pcs) + center_rep

        center_rep = center.unsqueeze(2).repeat(1, 1, 8, 1)
        quat_rep = quat.unsqueeze(2).repeat(1, 1, 8, 1)
        part_boxes_pred = qrot(quat_rep, box_vertex) + center_rep
        part_boxes_pred = box_12_lines(part_boxes_pred)

        center = gt_poses[:, :, :3]
        quat = gt_poses[:, :, 3:]
        center_rep = center.unsqueeze(2).repeat(1, 1, part_pcs.shape[2], 1)
        quat_rep = quat.unsqueeze(2).repeat(1, 1, part_pcs.shape[2], 1)
        part_pcs_gt = qrot(quat_rep, part_pcs) + center_rep

        center_rep = center.unsqueeze(2).repeat(1, 1, 8, 1)
        quat_rep = quat.unsqueeze(2).repeat(1, 1, 8, 1)
        part_boxes_gt = qrot(quat_rep, box_vertex) + center_rep
        part_boxes_gt = box_12_lines(part_boxes_gt)

        xyzrange=np.array([-1,1,-1,1,-1,1])
        for i in range(part_pcs.shape[0]):






            save_path = "/mnt/sda/vmware_share/assemble_result/" + str(shape_cd_loss.clone().detach().cpu().numpy()).ljust(18,'0')+ "_" + str(shape_id[i]).rjust(10,'0')+"_"+ str(part_valid[i].sum().cpu().numpy()).rjust(3,'0') + "_gt_pred.jpg"
            print(save_path)

            self.render_parts_pcs_pred_gt_in_one_image(part_pcs_pred[i], part_pcs_gt[i], part_boxes_pred[i], part_boxes_gt[i], save_path=save_path, part_ids=part_ids[i], xyzrange=xyzrange, show=True, \
                                                       shape_cd_loss=shape_cd_loss, shapecd_dist1=shapecd_dist1[i], shapecd_dist2=shapecd_dist2[i])

    def test_me(self):
        xyz = np.random.uniform(-10, 10, (2000, 3))
        feats = []
        feats.append(np.ones((len(xyz), 1)))
        feats = np.hstack(feats)

        voxel_size = 0.025

        coords = np.floor(xyz / voxel_size)
        _, unique_map, inverse_map = ME.utils.sparse_quantize(coords, return_index=True, return_inverse=True)
        inds = unique_map
        coords = coords[inds]
        return_coords = xyz[inds]
        coords = ME.utils.batched_coordinates([coords])

        feats = feats[inds]

        feats = torch.tensor(feats, dtype=torch.float32)
        coords = coords.to(dtype=torch.int32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

        xx = torch.ones((4, 480, 640)).cuda()
        bs, ys, xs = torch.where(xx > 0)
        print("test okay")


    def feature_extract_minkowski(self, point_clouds):


        if self.use_color:
            if self.xyz_color:
                coordinates, features = ME.utils.batch_sparse_collate(
                    [(p[:, :3] / self.voxel_size, p[:, :]) for p in point_clouds])
            else:
                coordinates, features = ME.utils.batch_sparse_collate(
                    [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in point_clouds])
        else:

            coordinates, features = ME.utils.batch_sparse_collate(

                [(p[:, :3] / self.voxel_size, p[:, :3]) for p in point_clouds])


        origin_voxel = ME.SparseTensor(coordinates=coordinates, features=features)

        x = self.pre_encoder(origin_voxel)
        batch_num = origin_voxel.C[:, 0].max().long() + 1




















        out = x[-1]
        features = out.F
        xyz = out.C[:, 1:] * self.voxel_size

        sampled_features_batch = []
        sampled_xyz_batch = []
        sample_inds_batch = []
        feature_pool_batch = []
        for batch_id in range(batch_num):
            batch_id_list = out.C[:, 0]
            batch_indices = torch.where(batch_id_list == batch_id)


            features_batch = features[batch_indices]
            xyz_batch = xyz[batch_indices]


            features_batch = features_batch.transpose(1, 0).contiguous()



            xyz_batch_squ = xyz_batch.unsqueeze(0)
            features_batch_squ = features_batch.unsqueeze(0)

            if self.random_fps:
                new_idx = torch.randperm(xyz_batch_squ.shape[1])

                features_batch_squ = features_batch_squ[:, :, new_idx]
                xyz_batch_squ = xyz_batch_squ[:, new_idx, :]









            features_pool = self.avgpool(features_batch_squ)



            feature_pool_batch.append(features_pool)



        enc_features_pool = torch.cat(feature_pool_batch)












        return enc_features_pool