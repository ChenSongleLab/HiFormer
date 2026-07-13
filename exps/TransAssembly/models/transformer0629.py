
"""
Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .models import MLP, Predictor, MLPRelu, PredictorBox

from model.swin3d_transformer import SwinTransformerBlock,grid_sample


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate=True, args=None
                 ):
        super().__init__()
        self.args = args



        encoder_layer = SwinTransformerBlock(d_model, nhead, 2, 0.1,
            rel_query=True, rel_key=True, rel_value=True, drop_path=dropout,\
            mlp_ratio=4.0, qkv_bias=True)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        encoder_layer_std = TransformerEncoderStandardLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, args=args)
        self.encoder = TransformerEncoder(encoder_layer, encoder_layer_std, num_encoder_layers, encoder_norm,
                                          return_intermediate=return_intermediate, args=args)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, **kwargs):

        src = src.permute(1, 0, 2)
        mask = ~mask.bool()

        tgt = torch.zeros_like(src)
        preds, memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, **kwargs)


        return preds, memory

class TDecoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate=True, args=None):
        super().__init__()
        self.args = args

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args=args)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate, args=args)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, mask, query_embed, pos_embed, **kwargs):

        tgt = tgt.transpose(0, 1)
        mask = ~mask.bool()

        pred_pose, pred_cate = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                            pos=pos_embed, query_pos=query_embed, **kwargs)

        return pred_pose.permute(2, 0, 1, 3), pred_cate.permute(2, 0, 1, 3)




class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, encoder_layer_std, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.args = args
        self.parts_layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


        d_model = encoder_layer.dim
        dim_in = d_model
        if args.pose_cat_in_encoder:
            dim_in += 7
        if args.noise_cat_in_encoder:
            dim_in += args.noise_dim
        if args.ins_cat_in_encoder:
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in += args.max_num_part * 2
        self.proj_encoder_in = nn.Linear(dim_in, d_model)


        d_model = encoder_layer.dim
        dim_in = d_model
        if args.base_cat:
            dim_in = dim_in + args.feat_dim
        if args.pose_cat:
            dim_in = dim_in + 7
        if args.noise_cat:
            dim_in = dim_in + args.noise_dim
        if args.ins_cat:
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in = dim_in + args.max_num_part * 2
        if args.group_tm_pred_with_higher_box:
            dim_in = dim_in + 12

        mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)

        predictor = Predictor(args.feat_dim)
        if not args.shared_pred:
            self.mlp = _get_clones(mlp, num_layers)
            self.predictor = _get_clones(predictor, num_layers)
        else:
            self.mlp = mlp
            self.predictor = predictor

        if args.group_tm:

            self.in_group_layers = _get_clones(encoder_layer_std, num_layers)
            self.out_group_layers = _get_clones(encoder_layer_std, num_layers)
            self.group_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.shape_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.group_part_pos_embed = nn.Parameter(torch.zeros(args.max_num_part_in_group,  d_model))

            pred_box = PredictorBox(d_model, d_model, 6, 3)
            self.mlp_box_part = nn.ModuleList([pred_box for _ in range(num_layers)])
            self.mlp_box_group = nn.ModuleList([pred_box for _ in range(num_layers)])
            self.mlp_box_shape = nn.ModuleList([pred_box for _ in range(num_layers)])

            if args.group_tm_cat_higher_emd:
                merge_emd = MLPRelu(d_model * 3, d_model, d_model, 2)
                self.proj_merge_emd = nn.ModuleList([merge_emd for _ in range(num_layers)])

            if args.group_tm_cat_higher_box:
                merge_box = MLPRelu(d_model + 12, d_model, d_model, 2)
                self.proj_merge_box = nn.ModuleList([merge_box for _ in range(num_layers)])

    def get_offset_index(self, parts_valid):
        N = int(parts_valid.sum())
        xyz =  torch.zeros((N, 3),dtype=torch.float32).to(parts_valid.device)
        window_size = 0.1
        window_size = torch.tensor([window_size] * 3,dtype=torch.float32).to(xyz.device)
        count = 0
        offset = []
        for i in range(int(parts_valid.shape[0])):
                count += int(parts_valid[i].sum())
                offset.append(count)
        offset = torch.IntTensor(offset).cuda()
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long().cuda()

        v2p_map, p2v_map, counts = grid_sample(xyz, batch, window_size, start=None)






        n, k = p2v_map.shape
        mask = torch.arange(k).unsqueeze(0).cuda() < counts.unsqueeze(-1)
        mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2))
        index_0 = p2v_map.unsqueeze(-1).expand(-1, -1, k)[
            mask_mat]
        index_1 = p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]
        M = index_0.shape[0]




        index_0, indices = torch.sort(index_0)
        index_1 = index_1[indices]
        index_0_counts = index_0.bincount()
        n_max = index_0_counts.max()
        index_0_offsets = index_0_counts.cumsum(dim=-1)
        index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0)

        assert index_0.shape[0] == index_1.shape[0]
        assert index_0.shape[0] == (counts ** 2).sum()




        return index_0, index_0_offsets, n_max, index_1

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        num_part, batch_size, dim = src.size()


        part_ids = kwargs["part_ids"]
        num_group = kwargs["num_group"]
        num_gpart = kwargs["num_group_part"]
        parts_valid_in_group = kwargs["parts_valid_in_group"]
        groups_valid = kwargs["groups_valid"]
        flat2group_index = kwargs["flat2group_index"]
        group2flat_index = kwargs["group2flat_index"]


        dense2flat_index = kwargs["dense2flat_index"]

        parts_valid = (~src_key_padding_mask).int()
        index_0, index_0_offsets, n_max, index_1 = self.get_offset_index(parts_valid)


        if self.args.noise_cat or self.args.noise_cat_in_encoder:
            assert self.args.train_mon > 1 or self.args.eval_mon > 1
            random_noise = torch.normal(mean=0., std=1., size=(num_part, batch_size, self.args.noise_dim)).to(src.device)

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
            ins_codes = ins_codes.transpose(0, 1).contiguous()

        if self.args.group_tm:

            in_group_src_key_padding_mask = ~parts_valid_in_group.bool()
            group_part_pos_embed = self.group_part_pos_embed[:num_gpart, ...].unsqueeze(0).repeat(batch_size * num_group, 1, 1)
            group_token = self.group_token.expand(batch_size * num_group, -1, -1)
            in_group_mask = in_group_src_key_padding_mask.view(batch_size * num_group,
                                                               num_gpart)
            in_group_token_mask = group_token.new_zeros(group_token.shape[:-1]).bool()
            in_group_mask = torch.cat((in_group_token_mask, in_group_mask), dim=1)


            out_group_src_key_padding_mask = ~groups_valid.bool()
            out_group_token_mask = out_group_src_key_padding_mask.new_zeros(batch_size, 1).bool()
            out_group_mask = torch.cat((out_group_token_mask, out_group_src_key_padding_mask), dim=1)
            shape_token = self.shape_token.expand(batch_size, -1, -1)
            shape_token = shape_token.permute(1, 0, 2)



            flat2group_index_exp = flat2group_index[:, :num_group, :num_gpart].reshape(batch_size, num_group * num_gpart)
            flat2group_index_exp = flat2group_index_exp.unsqueeze(-1).repeat(1, 1, dim)
            part_valids_group_mask = parts_valid_in_group.unsqueeze(-1).repeat(1, 1, 1, dim)


            group2flat_index_1d = group2flat_index[..., 0] * num_gpart + group2flat_index[..., 1]
            group2flat_index_exp = group2flat_index_1d.unsqueeze(-1).repeat(1, 1, dim)
            parts_valid_mask = parts_valid.unsqueeze(-1).repeat(1, 1, dim)


        output = src
        intermediate = []
        intermediate_part_pose = []
        intermediate_part_box = []
        intermediate_group_box = []
        intermediate_shape_box = []

        for idx, (parts_layer, in_group_layer, out_group_layer) in enumerate(zip(self.parts_layers, self.in_group_layers, self.out_group_layers)):

            if idx == 0:
                pred = src.new_zeros((num_part, batch_size, 7))
                xyz_flat = pred.permute(1, 0, 2)[:, :, 0:3]
            if self.args.pose_cat_in_encoder:
                output = torch.cat((output, pred), dim=-1)
            if self.args.noise_cat_in_encoder:
                output = torch.cat((output, random_noise), dim=-1)
            if self.args.ins_cat_in_encoder:
                output = torch.cat((output, ins_codes), dim=-1)
            output = self.proj_encoder_in(output)


            output = output.permute(1, 0, 2)
            feats = []
            xyz = []
            for i in range(batch_size):
                feats.append(output[i][:int(parts_valid[i].sum())])
                xyz.append(xyz_flat[i][:int(parts_valid[i].sum())])
            feats = torch.cat(feats)
            xyz = torch.cat(xyz)



            output = parts_layer(feats, xyz, index_0, index_0_offsets, n_max, index_1, 0)
            output = torch.gather(output, dim=0, index=dense2flat_index)
            output = output.view(batch_size, num_part, -1)
            if self.norm is not None:
                output = self.norm(output)


            if self.args.group_tm:






                out_group = torch.gather(output, dim=1, index=flat2group_index_exp)
                out_group = out_group.view(batch_size, num_group, num_gpart, dim)
                out_group = out_group * part_valids_group_mask

                out_group = out_group.reshape(batch_size * num_group, num_gpart, dim)
                out_group = out_group + group_part_pos_embed
                out_group = torch.cat((group_token, out_group), dim=1)
                out_group = out_group.permute(1, 0, 2)
                out_group = in_group_layer(out_group, src_mask=mask, src_key_padding_mask=in_group_mask, pos=pos)
                if self.norm is not None:
                    out_group = self.norm(output)
                out_group = out_group.permute(1, 0, 2)


                groups_embed = out_group.view(batch_size, num_group, num_gpart + 1, dim)[:, :, 0]
                groups_embed = groups_embed.permute(1, 0, 2)

                groups_embed_ext = torch.cat((shape_token, groups_embed), dim=0)
                groups_embed_ext = out_group_layer(groups_embed_ext, src_mask=mask, src_key_padding_mask=out_group_mask, pos=pos)
                if self.norm is not None:
                    groups_embed_ext = self.norm(groups_embed_ext)


                shape_embed = groups_embed_ext[:1, ...]
                pred_box_shape = self.mlp_box_shape[idx](shape_embed).view(batch_size, -1)
                shape_embed_cp = shape_embed.permute(1, 0, 2)

                shape_embed_cp = shape_embed_cp.unsqueeze(1).repeat(1, num_group, num_gpart, 1).reshape(
                    batch_size, num_group*num_gpart, dim)
                shape_box_cp = pred_box_shape.view(1, batch_size, -1).permute(1, 0, 2)
                shape_box_cp = shape_box_cp.unsqueeze(1).repeat(1, num_group, num_gpart, 1).reshape(
                    batch_size, num_group*num_gpart, -1)


                groups_embed = groups_embed_ext[1:, ...]
                groups_embed = groups_embed.permute(1, 0, 2)
                pred_box_group = self.mlp_box_group[idx](groups_embed)

                groups_embed_cp = groups_embed.unsqueeze(2).repeat(1, 1, num_gpart, 1).view(batch_size, num_group*num_gpart, -1)
                groups_box_cp = pred_box_group.unsqueeze(2).repeat(1, 1, num_gpart, 1).view(batch_size, num_group*num_gpart, -1)



                output_shape_emd = torch.gather(shape_embed_cp, dim=1, index=group2flat_index_exp)
                output_shape_emd = output_shape_emd * parts_valid_mask
                output_shape_emd = output_shape_emd.permute(1, 0, 2)
                output_shape_box = torch.gather(shape_box_cp, dim=1, index=group2flat_index_exp[:,:,:6])
                output_shape_box = output_shape_box * parts_valid_mask[:,:,:6]
                output_shape_box = output_shape_box.permute(1, 0, 2)

                output_group_emd = torch.gather(groups_embed_cp, dim=1, index=group2flat_index_exp)
                output_group_emd = output_group_emd * parts_valid_mask
                output_group_emd = output_group_emd.permute(1, 0, 2)
                output_group_box = torch.gather(groups_box_cp, dim=1, index=group2flat_index_exp[:,:,:6])
                output_group_box = output_group_box * parts_valid_mask[:,:,:6]
                output_group_box = output_group_box.permute(1, 0, 2)

                out_group = out_group[:, 1:, :]
                out_group = out_group.reshape(batch_size, num_group * num_gpart, dim)
                output = torch.gather(out_group, dim=1, index=group2flat_index_exp)
                output = output * parts_valid_mask
                output = output.permute(1, 0, 2)

                if self.args.group_tm_cat_higher_emd:
                    output_merge_emd = torch.cat([output, output_group_emd, output_shape_emd], dim=-1)
                    output = self.proj_merge_emd[idx](output_merge_emd)
                if self.args.group_tm_cat_higher_box:
                    output_merge_box = torch.cat([output, output_group_box, output_shape_box], dim=-1)
                    output = self.proj_merge_box[idx](output_merge_box)





            feat = output.clone()
            if self.args.group_tm_pred_with_higher_box:
                feat = torch.cat([feat, output_group_box, output_shape_box], dim=-1)
            if self.args.base_cat:
                feat = torch.cat((feat, src), dim=-1)
            if self.args.pose_cat:
                feat = torch.cat((feat, pred), dim=-1)
            if self.args.noise_cat:
                feat = torch.cat((feat, random_noise), dim=-1)
            if self.args.ins_cat:
                feat = torch.cat((feat, ins_codes), dim=-1)

            if not self.args.shared_pred:
                feat = self.mlp[idx](feat)
                pred = self.predictor[idx](feat)
                pred_box = self.mlp_box_part[idx](feat)
            else:
                feat = self.mlp(feat)
                pred = self.predictor(feat)


            if self.return_intermediate:
                intermediate_part_pose.append(pred.permute(1,0,2))
                intermediate_part_box.append(pred_box.permute(1,0,2))
                intermediate_group_box.append(pred_box_group)
                intermediate_shape_box.append(pred_box_shape)

            if self.args.pred_detach:
                    with torch.no_grad():
                        pred = pred.detach()

        if self.return_intermediate:
            intermediate.append(torch.stack(intermediate_part_pose))
            intermediate.append(torch.stack(intermediate_part_box))
            intermediate.append(torch.stack(intermediate_group_box))
            intermediate.append(torch.stack(intermediate_shape_box))
            return intermediate, output
        else:
            return pred.unsqueeze(0), output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.args = args
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


        dim_in = decoder_layer.d_model
        if args.pose_cat_in_decoder_pred:
            dim_in = dim_in + 7
        if args.noise_cat_in_decoder_pred:
            dim_in = dim_in + args.noise_dim

        pose_mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
        pose_pred = Predictor(args.feat_dim)
        if self.args.cate_on:
            cate_mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
            cate_pred = nn.Linear(args.feat_dim, 1)
        if not args.shared_pred:
            self.pose_mlp = _get_clones(pose_mlp, num_layers)
            self.pose_pred = _get_clones(pose_pred, num_layers)
            if self.args.cate_on:
                self.cate_mlp = _get_clones(cate_mlp, num_layers)
                self.cate_pred = _get_clones(cate_pred, num_layers)
        else:
            self.pose_mlp = pose_mlp
            self.pose_pred = pose_pred
            if self.args.cate_on:
                self.cate_mlp = cate_mlp
                self.cate_pred = cate_pred

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        num_part, batch_size, _ = tgt.size()

        if self.args.noise_cat_in_decoder_pred or self.args.noise_cat_in_decoder_trans:
            assert self.args.train_mon > 1 or self.args.eval_mon > 1
            random_noise = torch.normal(mean=0., std=1., size=(num_part, batch_size, self.args.noise_dim)).to(tgt.device)


        decode_mask = kwargs["decode_mask"]
        if self.args.ins_cat_in_decoder:
            part_ids = kwargs["part_ids"].long()
            _, num_part_per_ins = part_ids.size()

            decode_ids = kwargs["decode_ids"].long().view(-1)
            decode_bs = torch.arange(batch_size).to(decode_ids.device)



            match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part_per_ins)
            match_v = part_ids.unsqueeze(1).repeat(1, num_part_per_ins, 1)
            match_codes = (match_h == match_v).float()
            valid_mask = memory_key_padding_mask.unsqueeze(2).repeat(1, 1, num_part_per_ins)
            match_codes[valid_mask] = 0.

            cate_codes = torch.arange(num_part_per_ins).unsqueeze(0).repeat(batch_size, 1).to(part_ids.device)
            ins_codes = part_ids.new_zeros((batch_size, num_part_per_ins, num_part_per_ins * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
            ins_codes[..., :num_part_per_ins][valid_mask] = 0.
            ins_codes[..., num_part_per_ins:] = match_codes

            if num_part_per_ins < self.args.max_num_part:
                num_res = self.args.max_num_part - num_part_per_ins
                ins_codes_pad = ins_codes.new_zeros((batch_size, num_part_per_ins, self.args.max_num_part * 2))
                ins_codes_pad[..., :num_part_per_ins] = ins_codes[..., :num_part_per_ins]
                ins_codes_pad[..., self.args.max_num_part:-num_res] = ins_codes[..., num_part_per_ins:]
                ins_codes = ins_codes_pad.clone()
            ins_codes = ins_codes.transpose(0, 1).contiguous()


            decode_codes = ins_codes[decode_ids, decode_bs, :].view(num_part, batch_size, self.args.max_num_part * 2).contiguous()
            memory_codes = ins_codes[decode_mask.repeat(1, 1, self.args.max_num_part * 2)].\
                view(num_part_per_ins - num_part, batch_size, self.args.max_num_part * 2).contiguous()
        decode_mask_memory = decode_mask.squeeze(-1).transpose(0, 1).contiguous()
        memory_key_padding_mask = memory_key_padding_mask[decode_mask_memory].view(batch_size, -1).contiguous()

        if self.args.pose_cat_in_memory:
            memory_poses = kwargs["memory_poses"]
        if self.args.noise_cat_in_memory:
            num_memory, _, len_memory = memory.size()
            memory_noise = torch.normal(mean=0., std=1., size=(num_memory, batch_size, self.args.noise_dim)).to(memory.device)

        output = tgt
        pose_inters = []
        cate_inters = []

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                pred = tgt.new_zeros((num_part, batch_size, 7))


            if self.args.pose_cat_in_decoder_trans:
                output = torch.cat((output, pred), dim=-1)
            if self.args.noise_cat_in_decoder_trans:
                output = torch.cat((output, random_noise), dim=-1)


            if self.args.pose_cat_in_memory:
                memory = torch.cat((memory, memory_poses), dim=-1)
            if self.args.noise_cat_in_memory:
                memory = torch.cat((memory, memory_noise), dim=-1)

            if self.args.ins_cat_in_decoder:
                output = torch.cat((output, decode_codes), dim=-1)
                memory = torch.cat((memory, memory_codes), dim=-1)


            output, memory = layer(output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos)

            if self.norm is not None:
                output = self.norm(output)
            feat = output.clone()

            if self.args.pose_cat_in_decoder_pred:
                feat = torch.cat((feat, pred), dim=-1)
            if self.args.noise_cat_in_decoder_pred:
                feat = torch.cat((feat, random_noise), dim=-1)

            if not self.args.shared_pred:
                pose_feat = self.pose_mlp[idx](feat)
                pose_pred = self.pose_pred[idx](pose_feat)
                if self.args.cate_on:
                    cate_feat = self.cate_mlp[idx](feat)
                    cate_pred = self.cate_pred[idx](cate_feat)
            else:
                pose_feat = self.pose_mlp(feat)
                pose_pred = self.pose_pred(pose_feat)
                if self.args.cate_on:
                    cate_feat = self.cate_mlp(feat)
                    cate_pred = self.cate_pred(cate_feat)

            if self.return_intermediate:
                pose_inters.append(pose_pred)
                if self.args.cate_on:
                    cate_inters.append(cate_pred)
                else:
                    cate_inters.append(pose_pred.new_zeros(pose_pred.size()))

            if self.args.pred_detach:
                with torch.no_grad():
                    pred = pose_pred.detach()

        if self.return_intermediate:
            return torch.stack(pose_inters), torch.stack(cate_inters)
        else:
            return pose_inters.unsqueeze(0), cate_inters.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.args = args
        dim_in = d_model
        if args.pose_cat_in_encoder:
            dim_in += 7
        if args.noise_cat_in_encoder:
            dim_in += args.noise_dim
        if args.ins_cat_in_encoder:
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in += args.max_num_part * 2
        self.proj = nn.Linear(dim_in, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = self.proj(src)
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderStandardLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.args = args
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.args = args
        dim_in = d_model
        if args.pose_cat_in_decoder_trans:
            dim_in += 7
        if args.noise_cat_in_decoder_trans:
            dim_in += args.noise_dim

        dim_memory = d_model
        if args.pose_cat_in_memory:
            dim_memory += 7
        if args.noise_cat_in_memory:
            dim_memory += args.noise_dim

        if args.ins_cat_in_decoder:
            dim_in += args.max_num_part * 2
            dim_memory += args.max_num_part * 2
        self.proj = nn.Linear(dim_in, d_model)
        self.memory_proj = nn.Linear(dim_memory, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = self.proj(tgt)
        memory = self.memory_proj(memory)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos), memory
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos), memory


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_encoder(args):
    return TEncoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate=True,
        args=args
    )
True

def build_transformer_decoder(args):
    return TDecoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate=True,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
