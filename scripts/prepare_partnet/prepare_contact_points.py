

import os
import json
import copy
import numpy as np
import ipdb
import torch
from exps.utils.quaternion import qrot
import math
import random


def cal_distance(a, b):
    num_points = a.shape[0]
    a = torch.tensor(a)
    b = torch.tensor(b)
    A = a.unsqueeze(0).repeat(num_points, 1, 1)
    B = b.unsqueeze(1).repeat(1, num_points, 1)
    C = (A - B)**2
    C = np.array(C).sum(axis=2)
    ind = C.argmin()
    R_ind = ind // 1000
    C_ind = ind - R_ind * 1000
    return C.min(), C_ind


def get_pair_list(pts):
    delta1 = 1e-3
    cnt1 = 0
    num_part = pts.shape[0]
    connect_list = np.zeros((num_part, num_part, 4))

    for i in range(0, num_part):
        for j in range(0, num_part):
            if i == j: continue
            dist, point_ind = cal_distance(pts[i], pts[j])
            point = pts[i, point_ind]
            if dist < delta1:
                connect_list[i][j][0] = 1
                connect_list[i][j][1] = point[0]
                connect_list[i][j][2] = point[1]
                connect_list[i][j][3] = point[2]
            else:
                connect_list[i][j][0] = 0
                connect_list[i][j][1] = point[0]
                connect_list[i][j][2] = point[1]
                connect_list[i][j][3] = point[2]
    return connect_list


def find_pts_ind(part_pts, point):
    for i in range(len(part_pts)):
        if part_pts[i, 0] == point[0] and part_pts[i, 1] == point[1] and part_pts[i, 2] == point[2]:
            return i
    return -1


if __name__ == "__main__":
    root_to_load_file = "/data00/home/zhangrufeng1/projects/3DPartAssembly/data/partnet_dataset"
    root_to_load_data = "/data00/home/zhangrufeng1/projects/3DPartAssembly/prepare_data/shape_data"
    root_to_save_file = "/data00/home/zhangrufeng1/datasets/partnet/zhang_partnet/contact_points"
    cat_name = "Chair"
    modes = ['train', 'val', 'test']
    levels = [3, 2, 1]
    if not os.path.isfile(root_to_save_file):
        os.makedirs(root_to_save_file, exist_ok=True)

    for level in levels:
        for mode in modes:
            object_file = os.path.join(root_to_load_file, "stats/train_val_test_split", cat_name + "." + mode + ".json")
            object_json = json.load(open(object_file))
            object_list = [int(object_json[i]['anno_id']) for i in range(len(object_json))]
            idx = 0
            for id in object_list:
                idx += 1
                print("Convert {}/{}: level-{}-mode-{}-id-{}".format(idx, len(object_list), level, mode, id))
                if True:
                    cur_data_fn = os.path.join(root_to_load_data, '%s_level' % id + str(level) + '.npy')
                    cur_data = np.load(cur_data_fn, allow_pickle=True).item()
                    cur_pts = cur_data['part_pcs']
                    class_index = cur_data['part_ids']
                    num_point = cur_pts.shape[1]
                    poses = cur_data['part_poses']
                    quat = poses[:, 3:]
                    center = poses[:, :3]
                    gt_pts = copy.copy(cur_pts)
                    for i in range(len(cur_pts)):
                        gt_pts[i] = qrot(torch.from_numpy(quat[i]).unsqueeze(0).repeat(num_point, 1).unsqueeze(0),
                                         torch.from_numpy(cur_pts[i]).unsqueeze(0))
                        gt_pts[i] = gt_pts[i] + center[i]

                    oldfile = get_pair_list(gt_pts)
                    newfile = oldfile
                    for i in range(len(oldfile)):
                        for j in range(len(oldfile[0])):
                            if i == j: continue
                            point = oldfile[i, j, 1:]
                            ind = find_pts_ind(gt_pts[i], point)
                            if ind == -1:
                                ipdb.set_trace()
                            else:
                                newfile[i, j, 1:] = cur_pts[i, ind]
                    np.save(root_to_save + "contact_points/" + 'pairs_with_contact_points_%s_level' % id + str(level) + '.npy', newfile)
