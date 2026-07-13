
"""
    PartNetPartDataset
"""

import os
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
from quaternion import qrot

DATA_FEATURES = ['part_pcs', 'part_poses', 'part_valids', 'match_ids', 'contact_points', 'sym',
                 'shape_id', 'part_ids', 'pairs', 'pcs_group','parts_valid_in_group','part_poses_group',\
                 'groups_valid','part_in_group_index','gt_group_box','gt_shape_box', 'input_part_box', 'instance_code_group', 'num_group', \
                 'num_part_in_group', "flat2group_index", "group2flat_index", "instance_code"]

class PartNetPartDataset(data.Dataset):
    def __init__(self, data_dir, data_fn, category, data_features=DATA_FEATURES, level=3, max_num_part=20, max_num_part_in_group=13):
        self.data_dir = data_dir       
        self.data_fn = data_fn         
        self.category = category

        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part - 1) / 2
        self.level = level

       
        self.data = np.load(os.path.join(self.data_dir, data_fn))

       
        self.data_features = data_features

        self.part_sems = []
        self.part_sem2id = dict()


        self.max_num_part_in_group = max_num_part_in_group




    def get_part_count(self):
        return len(self.part_sems)
        
    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_fn: %s, max_num_part: %d' % \
                (self.category, len(self), self.data_dir, self.data_fn, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_id = self.data[index]

        cur_data_fn = os.path.join(self.data_dir, 'shape_data/%s_level' % shape_id + self.level + '.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        cur_contact_data_fn = os.path.join(self.data_dir, 'contact_points/pairs_with_contact_points_%s_level'
                                           % shape_id + self.level + '.npy')
        cur_contacts = np.load(cur_contact_data_fn, allow_pickle=True)

        cur_pts = cur_data['part_pcs']
        cur_pose = cur_data['part_poses']
        cur_geo_part_ids = np.array(cur_data['geo_part_ids'])
        cur_sym = cur_data['sym']
        cur_num_part = cur_pts.shape[0]



        if cur_num_part > self.max_num_part:
            return None
        part_valids = np.zeros((self.max_num_part,), dtype=np.float32)
        part_valids[:cur_num_part] = 1
        part_ids = np.zeros((self.max_num_part,), dtype=np.float32)
        part_ids[:cur_num_part] = cur_geo_part_ids + 1 

        num_group, num_part_in_group, cur_same_class_dict = self.split_group_data(part_valids, part_ids)
        gt_part_box, gt_group_box, gt_shape_box = self.get_boxes(cur_pts, cur_same_class_dict, cur_pose)
        input_part_box, _ , _ = self.get_boxes(cur_pts, cur_same_class_dict, cur_pose=None)
        inst_code = self.get_instance_code(part_valids, part_ids)
        flat2group_index, group2flat_index = self.get_conv_index(cur_same_class_dict)
        data_feats = ()
        for feat in self.data_features:
            if feat == 'contact_points':  
                out = np.zeros((self.max_num_part, self.max_num_part, 4), dtype=np.float32)
                out[:cur_num_part, :cur_num_part, :] = cur_contacts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_pcs': 
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
                out[:cur_num_part] = cur_pts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_poses': 
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, 3 + 4), dtype=np.float32)
                out[:cur_num_part] = cur_pose
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'semantic_ids': 
                cur_part_ids = cur_data['part_ids']
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_ids':
                if cur_num_part > self.max_num_part:
                    return None
                mapped_part_ids = cur_geo_part_ids + 1
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = mapped_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_valids':
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = 1.
                out = torch.from_numpy(out).float().unsqueeze(0)  
                data_feats = data_feats + (out,)

            elif feat == 'sym': 
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, cur_sym.shape[1]), dtype=np.float32)
                out[:cur_num_part] = cur_sym
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'pairs':
                if cur_num_part > self.max_num_part:
                    return None
                valid_pair_matrix = np.ones((cur_num_part, cur_num_part))
                pair_matrix = np.zeros((self.max_num_part, self.max_num_part))
                pair_matrix[:cur_num_part, :cur_num_part] = valid_pair_matrix
                out = torch.from_numpy(pair_matrix).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'match_ids': 
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                mapped_part_ids = cur_geo_part_ids + 1
                out[:cur_num_part] = mapped_part_ids
                map_id = 1
                for i in range(1, self.max_num_part + 1):
                    if i > mapped_part_ids.max():
                        break
                    idx = np.where(out == i)[0]
                    idx = torch.from_numpy(idx)
                    if len(idx) == 0:
                        continue
                    elif len(idx) == 1:
                        out[idx] = 0
                    else:
                        out[idx] = map_id
                        map_id += 1
                data_feats = data_feats + (out,)
            elif feat == 'pcs_group':
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, self.max_num_part_in_group, cur_pts.shape[1], 3), dtype=np.float32)
                for k, v in cur_same_class_dict.items():
                    inst_num = len(cur_same_class_dict[k])
                    out[k - 1, :inst_num] = cur_pts[v]
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'parts_valid_in_group':
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, self.max_num_part_in_group), dtype=np.float32)
                for k, v in cur_same_class_dict.items():
                    inst_num = len(cur_same_class_dict[k])
                    out[k - 1, :inst_num] = part_valids[v]
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'part_poses_group':
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, self.max_num_part_in_group, 7), dtype=np.float32)
                for k, v in cur_same_class_dict.items():
                    inst_num = len(cur_same_class_dict[k])
                    out[k - 1, :inst_num] = cur_pose[v]
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'groups_valid': 
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part), dtype=np.float32)
                for k, v in cur_same_class_dict.items():
                    out[k - 1] = 1 
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'part_in_group_index': 
                if cur_num_part > self.max_num_part:
                    return None
                data_feats = data_feats + (cur_same_class_dict,)
            elif feat == 'num_group':  
                if cur_num_part > self.max_num_part:
                    return None
                data_feats = data_feats + (num_group,)
            elif feat == 'num_part_in_group':  
                if cur_num_part > self.max_num_part:
                    return None
                data_feats = data_feats + (num_part_in_group,)
            elif feat == 'gt_group_box':
                data_feats = data_feats + (gt_group_box.unsqueeze(0),)
            elif feat == 'gt_shape_box':
                data_feats = data_feats + (gt_shape_box.unsqueeze(0),)
            elif feat == 'input_part_box':
                data_feats = data_feats + (input_part_box.unsqueeze(0),)
            elif feat == 'flat2group_index':
                data_feats = data_feats + (torch.from_numpy(flat2group_index).unsqueeze(0),)
            elif feat == 'group2flat_index':
                data_feats = data_feats + (torch.from_numpy(group2flat_index).unsqueeze(0),)
            elif feat == 'instance_code_group':
                out = np.zeros((self.max_num_part, self.max_num_part_in_group, 40), dtype=np.float32)
                for k, v in cur_same_class_dict.items():
                    inst_num = len(cur_same_class_dict[k])
                    out[k - 1, :inst_num] = inst_code[v]
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'instance_code':
                out = inst_code.unsqueeze(0)
                data_feats = data_feats + (out,)
            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

    def get_box(self, pts, cur_pose,  cur_same_class_dict):
        
        num_part, num_point, _ = pts.shape

        num_group = len(cur_same_class_dict)
        pts = torch.from_numpy(pts)
        center_tgt = torch.from_numpy(cur_pose[:, :3])
        quart_tgt = torch.from_numpy(cur_pose[:, 3:])
        center_tgt = center_tgt.unsqueeze(1).repeat(1, num_point, 1)
        pts_tgt = qrot(quart_tgt.unsqueeze(1).repeat(1, num_point, 1), pts) + center_tgt

        box_part_tgt_max = pts_tgt.view(-1, num_point, 3).max(1)[0]
        box_part_tgt_min = pts_tgt.view(-1, num_point, 3).min(1)[0]
        box_part_tgt =  torch.cat([(box_part_tgt_max + box_part_tgt_min)/2, box_part_tgt_max - box_part_tgt_min], dim=-1)

        gt_part_box_group = torch.zeros(self.max_num_part, 6) 
        gt_shape_box = torch.zeros(6)
        gt_shape_box_max = []
        gt_shape_box_min = []
        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            group_max = box_part_tgt_max.view(num_part, -1)[v].max(0)[0]
            group_min = box_part_tgt_min.view(num_part, -1)[v].min(0)[0]
            gt_part_box_group[k-1] = torch.cat([(group_max + group_min)/2, group_max - group_min], dim=-1)
            gt_shape_box_max.append(group_max)
            gt_shape_box_min.append(group_min)
        gt_shape_box_max = torch.stack(gt_shape_box_max).max(0)[0]
        gt_shape_box_min = torch.stack(gt_shape_box_min).min(0)[0]
        gt_shape_box = torch.cat([(gt_shape_box_max + gt_shape_box_min)/2, gt_shape_box_max - gt_shape_box_min], dim=-1)

        return gt_part_box_group, gt_shape_box

    def get_boxes(self, pts, cur_same_class_dict, cur_pose = None):
        
        num_part, num_point, _ = pts.shape
        num_group = len(cur_same_class_dict)
        pts = torch.from_numpy(pts)
        if not cur_pose is None:
            center_tgt = torch.from_numpy(cur_pose[:, :3])
            quart_tgt = torch.from_numpy(cur_pose[:, 3:])
            center_tgt = center_tgt.unsqueeze(1).repeat(1, num_point, 1)
            pts_tgt = qrot(quart_tgt.unsqueeze(1).repeat(1, num_point, 1), pts) + center_tgt
        else:
            pts_tgt = pts

        box_part_tgt_max = pts_tgt.view(-1, num_point, 3).max(1)[0]
        box_part_tgt_min = pts_tgt.view(-1, num_point, 3).min(1)[0]
        box_part = torch.zeros(self.max_num_part, 6)
        box_part[:num_part] =  torch.cat([(box_part_tgt_max + box_part_tgt_min)/2, box_part_tgt_max - box_part_tgt_min], dim=-1)

        box_group = torch.zeros(self.max_num_part, 6) 
        box_shape = torch.zeros(6)
        shape_box_max = []
        shape_box_min = []
        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            group_max = box_part_tgt_max.view(num_part, -1)[v].max(0)[0]
            group_min = box_part_tgt_min.view(num_part, -1)[v].min(0)[0]
            box_group[k-1] = torch.cat([(group_max + group_min)/2, group_max - group_min], dim=-1)
            shape_box_max.append(group_max)
            shape_box_min.append(group_min)
        shape_box_max = torch.stack(shape_box_max).max(0)[0]
        shape_box_min = torch.stack(shape_box_min).min(0)[0]
        box_shape = torch.cat([(shape_box_max + shape_box_min)/2, shape_box_max - shape_box_min], dim=-1)

        return box_part, box_group, box_shape

    def get_instance_code(self, part_valids, part_ids):
        
        num_part = self.max_num_part
        part_ids = torch.from_numpy(part_ids)
        part_valids = torch.from_numpy(part_valids)
        match_h = part_ids.unsqueeze(1).repeat(1, num_part)
        match_v = part_ids.unsqueeze(0).repeat(num_part, 1)
        match_codes = (match_h == match_v).float()  
        src_key_padding_mask = ~part_valids.bool()
        valid_mask = src_key_padding_mask.unsqueeze(1).repeat(1, num_part)
        match_codes[valid_mask] = 0.

        cate_codes = torch.arange(num_part)
        ins_codes = part_ids.new_zeros((num_part, num_part * 2)).scatter_(1, cate_codes.unsqueeze(1), 1)
        ins_codes[..., :num_part][valid_mask] = 0. 
        ins_codes[..., num_part:] = match_codes  
        return ins_codes
        
    def split_group_data(self, part_valids, part_ids):
        num_part_in_group = 0
        cur_same_class_dict = {}
        for j in range(int(part_valids.sum())):
            cur_class = int(part_ids[j])  
            if cur_class not in cur_same_class_dict:
                cur_same_class_dict[int(cur_class)] = []
            cur_same_class_dict[int(cur_class)].append(j)
            num_part_in_group = max(num_part_in_group, len(cur_same_class_dict[int(cur_class)]))
        num_group = len(cur_same_class_dict)  

        return num_group,num_part_in_group, cur_same_class_dict

    def get_conv_index(self, cur_same_class_dict):
        flat2group_index = np.zeros((self.max_num_part, self.max_num_part_in_group), dtype=np.int64)
        group2flat_index = np.zeros((self.max_num_part, 2), dtype=np.int64)  
        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            flat2group_index[k - 1, :inst_num] = v  
            for j in range(inst_num):
                group2flat_index[v[j]] = [(k - 1), j]  
        return flat2group_index, group2flat_index

        
    def test_conv_index(self, cur_pts, cur_same_class_dict,part_valids):

        part_valids_group = np.zeros((self.max_num_part, self.max_num_part_in_group), dtype=np.float32)
        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            part_valids_group[k - 1, :inst_num] = part_valids[v]
        part_valids_group = torch.from_numpy(part_valids_group).float()

        flat2group_index = np.zeros((self.max_num_part, self.max_num_part_in_group), dtype=np.int64)
        group2flat_index_2d = np.zeros((self.max_num_part, 2), dtype=np.int64)

        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            flat2group_index[k - 1, :inst_num] = v
            for j in range(inst_num):
                group2flat_index_2d[v[j]] = [(k-1),j]


        cur_pts = torch.from_numpy(cur_pts)
        flat2group_index = torch.from_numpy(flat2group_index)

        out_group_pcs = torch.zeros((self.max_num_part, self.max_num_part_in_group, cur_pts.shape[1], 3), dtype=torch.float32)
        for k, v in cur_same_class_dict.items():
            inst_num = len(cur_same_class_dict[k])
            out_group_pcs[k - 1, :inst_num] = cur_pts[v]

        flat2group_index_exp = flat2group_index.view(self.max_num_part*self.max_num_part_in_group)

        flat2group_index_exp = flat2group_index_exp.unsqueeze(-1).unsqueeze(-1).repeat(1, cur_pts.shape[-2],cur_pts.shape[-1])
        out_group_pcs2 = torch.gather(cur_pts, dim=0, index=flat2group_index_exp)
        out_group_pcs2 = out_group_pcs2.view(self.max_num_part, self.max_num_part_in_group, 1000, 3)
        part_valids_group_mask = part_valids_group.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, cur_pts.shape[-2],cur_pts.shape[-1])
        out_group_pcs3 = out_group_pcs2*part_valids_group_mask
        diff = out_group_pcs-out_group_pcs3





        group2flat_index = group2flat_index_2d[...,0]* self.max_num_part_in_group + group2flat_index_2d[...,1]
        group2flat_index = torch.from_numpy(group2flat_index)
        out_group_pcs4 = out_group_pcs3.view(self.max_num_part*self.max_num_part_in_group, 1000, 3)
        group2flat_index_exp = group2flat_index.unsqueeze(-1).unsqueeze(-1).repeat(1, out_group_pcs4.shape[-2],out_group_pcs4.shape[-1])

        curs_cov = torch.gather(out_group_pcs4, dim=0, index=group2flat_index_exp)
        part_valids_mask = torch.from_numpy(part_valids).unsqueeze(-1).unsqueeze(-1).repeat(1, cur_pts.shape[-2],cur_pts.shape[-1])
        curs_cov2 = curs_cov*part_valids_mask
        diff2 = cur_pts-curs_cov2[:cur_pts.shape[0]]
        return






if __name__ == "__main__":
    dataset = PartNetPartDataset(data_dir="/data00/home/zhangrufeng1/datasets/partnet/prepare_data", data_fn="Chair.train.npy",
                                 category="Chair", data_features=DATA_FEATURES, level='3', max_num_part=100)
    for data in dataset:
        aa = 1

