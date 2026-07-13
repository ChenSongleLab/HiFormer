
import copy
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MLP, Predictor


class MLP_2F(nn.Module):
    def __init__(self, feat_dim):
        super(MLP_2F, self).__init__()

        self.conv1 = nn.Conv1d(2 * feat_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_dim, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_dim)

    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):


        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class RelationPredictor(nn.Module):
    def __init__(self):
        super(RelationPredictor, self).__init__()
        self.mlp1 = nn.Linear(128 + 128, 256)
        self.mlp2 = nn.Linear(256, 512)
        self.mlp3 = nn.Linear(512, 1)

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        x = torch.relu(self.norm1(self.mlp1(x)))
        x = torch.relu(self.norm2(self.mlp2(x)))
        x = torch.sigmoid(self.mlp3(x))
        return x


class PoseExtractor(nn.Module):
    def __init__(self):
        super(PoseExtractor, self).__init__()
        self.mlp1 = nn.Linear(7, 256)
        self.mlp2 = nn.Linear(256, 128)

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x):
        x = torch.relu(self.norm1(self.mlp1(x)))
        x = torch.relu(self.norm2(self.mlp2(x)))
        return x


class AssemblyGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_iters = args.num_iters


        dim_in = args.feat_dim
        if args.pose_cat_in_encoder:
            dim_in = dim_in + 7
        if args.noise_cat_in_encoder:
            dim_in = dim_in + args.noise_dim
        if args.ins_cat_in_encoder:
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in = dim_in + args.max_num_part * 2

        proj = nn.Linear(dim_in, args.feat_dim)
        self.proj = _get_clones(proj, self.num_iters)

        extractor = MLP_2F(args.feat_dim)
        self.extractor_1 = _get_clones(extractor, self.num_iters)
        self.extractor_2 = _get_clones(extractor, self.num_iters)


        self.pose_extractor = PoseExtractor()
        self.relation_predictor = RelationPredictor()


        dim_in = args.feat_dim
        if args.base_cat:
            dim_in = dim_in + args.feat_dim
        if args.pose_cat:
            dim_in = dim_in + 7
        if args.noise_cat:
            dim_in = dim_in + args.noise_dim
        if args.ins_cat:
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in = dim_in + args.max_num_part * 2

        mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
        predictor = Predictor(args.feat_dim)
        if not args.shared_pred:
            self.mlp = _get_clones(mlp, self.num_iters)
            self.predictor = _get_clones(predictor, self.num_iters)
        else:
            self.mlp = mlp
            self.predictor = predictor

    def forward(self, base_feat, part_valid, **kwargs):
        batch_size, num_part, _ = base_feat.size()
        src_key_padding_mask = ~part_valid.bool()


        relation_h = part_valid.unsqueeze(2).repeat(1, 1, num_part)
        relation_v = part_valid.unsqueeze(1).repeat(1, num_part, 1)
        relation_codes = (relation_h == relation_v).float()
        valid_mask = src_key_padding_mask.unsqueeze(2).repeat(1, 1, num_part)
        relation_codes[valid_mask] = 0.
        valid_codes = copy.deepcopy(relation_codes)

        if self.args.noise_cat or self.args.noise_cat_in_encoder:
            assert self.args.train_mon > 1 or self.args.eval_mon > 1
            random_noise = torch.normal(mean=0., std=1., size=(batch_size, num_part, self.args.noise_dim)).to(base_feat.device)


        if self.args.ins_cat or self.args.ins_cat_in_encoder:
            part_ids = kwargs["part_ids"].long()

            if self.args.ins_version == "v1":

                match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
                match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
                match_codes = (match_h == match_v).float()
                valid_mask = src_key_padding_mask.unsqueeze(2).repeat(1, 1, num_part)
                match_codes[valid_mask] = 0.
                eye_mask = torch.eye(num_part).unsqueeze(0).repeat(batch_size, 1, 1).bool().to(part_ids.device)
                match_codes[eye_mask] = 0.

                inter_codes = (part_ids - (~src_key_padding_mask).long()).unsqueeze(2)
                if self.args.ins_cat_intra_only:
                    ins_codes = match_codes.clone()
                elif self.args.ins_cat_inter_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part)).scatter_(2, inter_codes, 1)
                    ins_codes[valid_mask] = 0.
                else:

                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, inter_codes, 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                    ins_codes[..., num_part:] = match_codes

            elif self.args.ins_version == "v2":

                match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
                match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
                match_codes = (match_h == match_v).float()
                valid_mask = src_key_padding_mask.unsqueeze(2).repeat(1, 1, num_part)
                match_codes[valid_mask] = 0.

                cate_codes = torch.arange(num_part).unsqueeze(0).repeat(batch_size, 1).to(part_ids.device)
                if self.args.ins_cat_inter_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                elif self.args.ins_cat_intra_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2))
                    ins_codes[..., num_part:] = match_codes
                else:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                    ins_codes[..., num_part:] = match_codes

            else:
                raise NotImplementedError

            if num_part < self.args.max_num_part:
                num_res = self.args.max_num_part - num_part
                ins_codes_pad = ins_codes.new_zeros((batch_size, num_part, self.args.max_num_part * 2))
                ins_codes_pad[..., :num_part] = ins_codes[..., :num_part]
                ins_codes_pad[..., self.args.max_num_part:-num_res] = ins_codes[..., num_part:]
                ins_codes = ins_codes_pad.clone()



        output = base_feat
        intermediate = []
        for iter_idx in range(self.num_iters):
            if iter_idx == 0:
                pred = base_feat.new_zeros((batch_size, num_part, 7))
            if self.args.pose_cat_in_encoder:
                output = torch.cat((output, pred), dim=-1)
            if self.args.noise_cat_in_encoder:
                output = torch.cat((output, random_noise), dim=-1)
            if self.args.ins_cat_in_encoder:
                output = torch.cat((output, ins_codes), dim=-1)


            output = self.proj[iter_idx](output)


            if iter_idx > 0:
                cur_pose = pred.clone()
                pose_feat = self.pose_extractor(cur_pose)
                pose_h = pose_feat.unsqueeze(2).repeat(1, 1, num_part, 1)
                pose_v = pose_feat.unsqueeze(1).repeat(1, num_part, 1, 1)
                rel_input = torch.cat((pose_h, pose_v), dim=-1)
                relation_up = self.relation_predictor(rel_input.view(batch_size, -1, 256)).view(batch_size, num_part, num_part)
                relation_codes = relation_up * valid_codes


            feat_h = output.unsqueeze(2).repeat(1, 1, num_part, 1)
            feat_v = output.unsqueeze(1).repeat(1, num_part, 1, 1)
            rel_input = torch.cat((feat_h, feat_v), dim=-1)
            part_relation = self.extractor_1[iter_idx](
                rel_input.view(batch_size * num_part, num_part, -1)).view(batch_size, num_part, num_part, -1)


            part_message = part_relation * relation_codes.unsqueeze(-1)
            part_message = part_message.sum(dim=2)
            norm = relation_codes.sum(dim=-1)
            delta = 1e-6
            normed_part_message = part_message / (norm.unsqueeze(dim=2) + delta)
            output = torch.cat((output, normed_part_message), dim=-1)
            output = self.extractor_2[iter_idx](output)

            feat = output.clone()
            if self.args.base_cat:
                feat = torch.cat((feat, base_feat), dim=-1)
            if self.args.pose_cat:
                feat = torch.cat((feat, pred), dim=-1)
            if self.args.noise_cat:
                feat = torch.cat((feat, random_noise), dim=-1)
            if self.args.ins_cat:
                feat = torch.cat((feat, ins_codes), dim=-1)

            if not self.args.shared_pred:
                feat = self.mlp[iter_idx](feat)
                pred = self.predictor[iter_idx](feat)
            else:
                feat = self.mlp(feat)
                pred = self.predictor(feat)

            intermediate.append(pred)

            if self.args.pred_detach:
                with torch.no_grad():
                    pred = pred.detach()

        preds = torch.stack(intermediate)

        return preds.permute(1, 0, 2, 3)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
