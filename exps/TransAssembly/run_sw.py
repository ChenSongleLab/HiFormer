

import argparse
import os
import os.path as osp
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from tensorboardX import  SummaryWriter

import matplotlib.pyplot as plt
plt.switch_backend('agg')
warnings.filterwarnings('ignore')

import random, numpy

from scripts import d_utils as d_utils
from scripts import z_utils as z_utils
from models.trans_assembly_encoder import TransAssembly_encoder,show_process_gpu_info
from models.trans_assembly_gnn import TransAssembly_gnn
from models.trans_assembly_freeze import TransAssembly_freeze
from datasets.partnet import PartNetPartDataset
from train3d import train_one_epoch
from eval3d import eval_func
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



parser = argparse.ArgumentParser(description='3D part assembly.')

print(torch.__version__)




parser.add_argument('--data_dir', type=str, default='/media/chensl/data/chensl/prepare_data', help='data directory')
parser.add_argument('--category', type=str, default='Chair', choices=['Chair', 'Table', 'Lamp'], help='model def file')
parser.add_argument('--train_data_fn', type=str, default='Chair.train.npy',
                    choices=['Chair.train.npy', 'Table.train.npy', 'Lamp.train.npy'],
                    help='training data file that index all data tuples')
parser.add_argument('--val_data_fn', type=str, default='Chair.val.npy',
                    choices=['Chair.val.npy', 'Table.val.npy', 'Lamp.val.npy'],
                    help='validation data file that index all data tuples')
parser.add_argument('--level', type=str, default='3', help='level of dataset')


parser.add_argument('--eval-only', default=0, type=int)
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--warmup-epochs', default=5, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--type-sched', default='cosine', type=str,
                    choices=['step', 'cosine'], help='The type of learning rate update.')
parser.add_argument('-b', '--batch-size', default=18   , type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='default initial (base) learning rate')
parser.add_argument('--lr_backbone', '--backbone-learning-rate', default=1.5e-4, type=float,
                    metavar='LRBackbone', help='backbone initial (base) learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')

parser.add_argument('--output-dir', default="/media/chensl/data/chensl/assemble_run/chair0817/swt_hie_multitask_box_fixbk_rand192_chair20250927/", type=str)
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 10)')
parser.add_argument('--resume', default="/media/chensl/data/chensl/assemble_run/chair0817/swt_hie_multitask_box_start0_norand_chair/checkpoint.pth.tar", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')



parser.add_argument('--group_tm',  default=1, type=int, metavar='N', help='0 for not use group transformer')
parser.add_argument('--max_num_part_in_group', type=int, default=13)
parser.add_argument('--group_tm_cat_higher_emd', type=int, default=1)
parser.add_argument('--group_tm_cat_higher_box', type=int, default=0)
parser.add_argument('--group_tm_pred_with_higher_box', type=int, default=0)




parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-eb', '--eval-batch-size', default=20, type=int, metavar='N')
parser.add_argument('--eval-epochs', default=1, type=int, metavar='N', help='epoch start for evaluation.')


parser.add_argument('--loss_weight_trans_l2', type=float, default=1.0, help='loss weight')
parser.add_argument('--loss_weight_rot_l2', type=float, default=0.0, help='loss weight')
parser.add_argument('--loss_weight_rot_cd', type=float, default=10.0, help='loss weight')
parser.add_argument('--loss_weight_shape_cd', type=float, default=1.0, help='loss weight')
parser.add_argument('--loss_weight_cate', type=float, default=0.0, help='loss weight')
parser.add_argument('--loss_weight_decode', type=float, default=1.0, help='loss weight')

parser.add_argument('--loss_weight_box', type=float, default=0.3, help='loss weight')
parser.add_argument('--loss_weight_group_cd', type=float, default=1.0, help='loss weight')
parser.add_argument('--loss_part_vertex_l12', type=float, default=0.0, help='loss weight part vertex，the box is rotate')


parser.add_argument('--model', default='trans_assembly_encoder', type=str, help='model name.')
parser.add_argument('--backbone', default='swin_transformer', type=str, help='pointnet_cls, minkowski, swin_transformer, point_voxel, the backbone to extract init features.')
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--max_num_part', type=int, default=20)
parser.add_argument('--num_mlp', type=int, default=2)
parser.add_argument('--base-cat', default=0, type=int, help='Whether to cat base feat when prediction.')
parser.add_argument('--pose-cat', default=0, type=int, help='Whether to cat pose when prediction.')
parser.add_argument('--pose-cat-in-encoder', default=0, type=int, help='Whether to cat pose in trans-encoder.')
parser.add_argument('--shared-pred', default=0, type=int, help='Whether to apply shared mlp & predictor.')
parser.add_argument('--pred-detach', default=1, type=int, help='Whether to detach pose when cat pose.')
parser.add_argument('--train-mon', default=5, type=int, help='MoN iterations in training.')
parser.add_argument('--eval-mon', default=10, type=int, help='MoN iterations in inference.')
parser.add_argument('--noise-cat', default=1, type=int, help='Whether to cat noise when prediction.')
parser.add_argument('--noise-cat-in-encoder', default=1, type=int, help='Whether to cat noise when in trans-encoder.')
parser.add_argument('--box-vertex-cat-in-encoder', default=1, type=int, help='Whether to cat noise when in trans-encoder.')
parser.add_argument('--embedding_pred_center_rotate', default=0, type=int, help='Whether to cat noise when in trans-encoder.')
parser.add_argument('--relative_box_position_encoding', default=1, type=int, help='Whether to cat noise when in trans-encoder.')
parser.add_argument('--cascade_pred', default=0, type=int, help='Whether to cat noise when in trans-encoder.')


parser.add_argument('--noise-dim', default=192, type=int, help='The dim of random noise.')
parser.add_argument('--ins-cat', default=0, type=int, help='Whether to cat ins part id when prediction.')
parser.add_argument('--ins-cat-in-encoder', default=1, type=int, help='Whether to cat ins part id when in trans-encoder.')
parser.add_argument('--ins-cat-inter-only', default=0, type=int, help='Whether to only cat ins inter-class encoding in trans-encoder.')
parser.add_argument('--ins-cat-intra-only', default=0, type=int, help='Whether to only cat ins intra-class encoding in trans-encoder.')
parser.add_argument('--ins-version', default='v2', type=str, help='the version of instance encoding.')
parser.add_argument('--type-eval', default='encoder', type=str, choices=['encoder', 'wip', 'decoder'], help='eval type.')
parser.add_argument('--worst-mon', default=0, type=int, help='Whether to apply worst mon during encoder inference.')



parser.add_argument('--stem_transformer', default=0, type=int, help='')
parser.add_argument('--use_xyz', default=1, type=int, help='')
parser.add_argument('--sync_bn', default=1, type=int, help='')
parser.add_argument('--rel_query', default=1, type=int, help='windows内部的是否用相对位置query编码')
parser.add_argument('--rel_key', default=1, type=int, help='windows内部的是否用相对位置key编码')
parser.add_argument('--rel_value', default=1, type=int, help='windows内部的是否用相对位置value编码')
parser.add_argument('--quant_size', default=0.005, type=float, help='在windows内部相对位置的量化大小, '
                                                                    '根据gridsize计算得到windowssize除以该值得到多少个量化单位，目前为10 ')
parser.add_argument('--num_layers', default=4, type=int, help='')
parser.add_argument('--patch_size', default=1, type=int, help='')
parser.add_argument('--window_size', default=5, type=int, help='')
parser.add_argument('--depths', default=[3, 3, 3, 3], type=list, help='swin transformerbasic中进行多少次的SwinTransformerBlock')
parser.add_argument('--channels', default=[32, 64, 128, 256], type=list, help='')
parser.add_argument('--num_heads', default= [1, 2, 4, 8], type=list, help='')
parser.add_argument('--up_k', default=3, type=int, help='向上采样的参数，暂时没有用到')
parser.add_argument('--drop_path_rate', default=0.3, type=float, help='')
parser.add_argument('--concat_xyz', default=0, type=int, help='默认是1，因为我们只有xyz坐标，所以设置为0')
parser.add_argument('--grid_size', default=0.01, type=int, help='根据该值计算window size')
parser.add_argument('--max_batch_points', default=250000, type=int, help='')
parser.add_argument('--max_num_neighbors', default=34, type=int, help='')
parser.add_argument('--ratio', default=0.5, type=float, help='每次downsample保留的百分比，也就是1/4')
parser.add_argument('--k', default=16, type=int, help='downsample的时候，每个点找到k个近邻，然后用线性变换融合该特征')
parser.add_argument('--classes', default=3, type=int, help='这个是和每个点对应的特征类似的，得到每个点的类别')




parser.add_argument('--decode-on', default=0, type=int, help='Whether to use transformer decoder.')
parser.add_argument('--num-pos', default=1, type=int, help='Number of positive query in decoder.')
parser.add_argument('--rand-pos', default=0, type=int, help='Whether to apply random positive number (1 - num-pos)')
parser.add_argument('--pose-cat-in-decoder-pred', default=1, type=int, help='Whether to cat pose when in decoder-pred.')
parser.add_argument('--pose-cat-in-decoder-trans', default=1, type=int, help='Whether to cat pose when in decoder-trans.')
parser.add_argument('--noise-cat-in-decoder-pred', default=0, type=int, help='Whether to cat noise when in decoder-pred.')
parser.add_argument('--noise-cat-in-decoder-trans', default=0, type=int, help='Whether to cat noise when in decoder-trans.')
parser.add_argument('--memory-detach', default=0, type=int, help='Whether to detach memory feats.')
parser.add_argument('--feat-in-detach', default=0, type=int, help='Whether to detach in-decoder feats.')
parser.add_argument('--cate-on', default=0, type=int, help='Whether to apply cate pred during decoder.')
parser.add_argument('--encode-freeze', default=0, type=int, help='Whether to freeze backbone/encoder during training.')
parser.add_argument('--ins-cat-in-decoder', default=1, type=int, help='Whether to cat ins part id in trans-decoder.')
parser.add_argument('--pose-cat-in-memory', default=0, type=int, help='Whether to cat pose in memory.')
parser.add_argument('--noise-cat-in-memory', default=0, type=int, help='Whether to cat noise in memory.')


parser.add_argument('--filter-on', default=0, type=int, help='Whether to apply filter augmentation.')
parser.add_argument('--filter-thresh', type=float, default=0.2, help='filter thresh')
parser.add_argument('--num-filter', default=1, type=int, help='Number of filter in training.')


parser.add_argument('--gt-vis', default=0, type=int, help='Whether to apply gt visualization.')
parser.add_argument('--gt-vis-dir', default='gt_vis/', type=str, help='relative path to root dir.')
parser.add_argument('--pred-encoder-vis', default=0, type=int, help='Whether to apply encoder pred visualization.')
parser.add_argument('--pred-encoder-vis-dir', default='encoder_pred/', type=str, help='relative path to root dir.')
parser.add_argument('--num-pred-vis', default=3, type=int, help='Number of pred visualization.')


parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=1, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')


parser.add_argument('--num-iters', default=5, type=int, help="Number of iteration layers in GNN")


parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--random_fps", default=False, type=bool)


parser.add_argument('--debug-on', default=0, type=int, help='Whether to apply gt visualization and set suffle of train_loader False.')


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def main():


    args = parser.parse_args()
    if args.debug_on:
        args.batch_size = 2
    d_utils.init_distributed_mode(args)
    print(args)

    z_utils.func_init(args)

    assert args.category in ["Chair", "Table", "Lamp"]
    if args.train_mon > 1 or args.eval_mon > 1:
        assert args.noise_cat or args.noise_cat_in_encoder, print("Noise issue 1...")
    else:
        assert (not args.noise_cat) and (not args.noise_cat_in_encoder), print("Noise issue 2...")





    seed = 42
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if args.model == "trans_assembly_encoder":
        model = TransAssembly_encoder(args)
        model = z_utils.device_func(model, args)
    elif args.model == "trans_assembly_freeze":
        model = TransAssembly_freeze(args)
        model = z_utils.device_func(model, args)
    elif args.model == "trans_assembly_gnn":
        model = TransAssembly_gnn(args)
        model = z_utils.device_func(model, args)
    else:
        raise NotImplementedError




    args.lr_backbone_names = ["swin_tm"]

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if (not match_name_keywords(n, args.lr_backbone_names)) and p.requires_grad],
            "lr": args.lr,
        }
        ,




    ]
    for n, p in model.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad:
            p.requires_grad = False

    if args.encode_freeze:
        assert args.decode_on, print("Decoder needed in this version.")
        if not args.eval_only:
            assert os.path.isfile(args.resume), print("Pre-train model needed in this version.")
        parameters = list()
        for name, param in model.named_parameters():
            if "decoder" in name:
                parameters.append(param)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    if args.type_sched == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    for t in param_dicts:
        print(f"sub number of params: { sum(p.numel() for p in t['params'])}")
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total number of params: {n_parameters}")

    scaler = torch.cuda.amp.GradScaler()


    output_dir = osp.join(osp.abspath(osp.dirname(__file__)), args.output_dir)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=output_dir) if args.rank == 0 else None
    log_writer = open(osp.join(output_dir, "train_log.txt"), "a")
    print(args)
    log_writer.write(str(args) + "\n")
    log_writer.flush()



    if args.resume:
        if os.path.exists(args.resume):
            model, run_epoch = z_utils.load_model(model, args, resume_on=True)
            args.start_epoch += run_epoch
    args.start_epoch = 0
    cudnn.benchmark = True


    if not args.eval_only:
        train_dataset = PartNetPartDataset(args.data_dir, args.train_data_fn, args.category,
                                           level=args.level, max_num_part=args.max_num_part, max_num_part_in_group=args.max_num_part_in_group)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        if args.debug_on:
            shuffle = False
        else:
            shuffle = True
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            drop_last=True, collate_fn=z_utils.collate_feats_with_none)

    val_dataset = PartNetPartDataset(args.data_dir, args.val_data_fn, args.category,
                                     level=args.level, max_num_part=args.max_num_part, max_num_part_in_group=args.max_num_part_in_group)
    val_sampler = None






    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        drop_last=False, collate_fn=z_utils.collate_feats_with_none)



    part_acc = 0

    if args.eval_only:
        part_acc_max = part_acc



        for i in range(100):
            part_acc = eval_func(val_loader, model, log_writer, summary_writer, args.start_epoch, args)
            if part_acc > part_acc_max:
                part_acc_max = part_acc
                print("i:", i, " part_acc_max", part_acc_max)
        return



    countdown = 0.
    best_cd = 1000
    best_acc = 0.
    best_conn = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)


        epoch_start_time = time.time()
        iter_loader = iter(train_loader)
        batch_finished = 0
        batch_finished_list = []




        countdown = train_one_epoch(train_loader, model, optimizer, scaler, summary_writer,
                                    log_writer, epoch, countdown, args)

        epoch_end_time = time.time()
        print("epoch time seconds ", epoch_end_time - epoch_start_time)
        if args.type_sched == "step":
            lr_scheduler.step()


        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            if (epoch + 1) >= args.eval_epochs:
                shape_cd, part_acc, contact_acc = eval_func(val_loader, model, log_writer, summary_writer, epoch, args)
            else:
                shape_cd = 1000
                part_acc = 0.
                contact_acc = 0.

            z_utils.save_model(epoch, scaler, output_dir, model, optimizer, part_acc, best_acc, args)
            best_cd = shape_cd if shape_cd < best_cd else best_cd
            best_acc = part_acc if part_acc > best_acc else best_acc
            best_conn = contact_acc if contact_acc > best_conn else best_conn
            print("==========================================================")
            print("Best Shape Chamfer: {}, Part Accuracy: {}, Part Connect: {}".format(best_cd, best_acc, best_conn))
            print("==========================================================")
            res_best = "\n==========================================================" + "\n" + \
                       "Best Shape Chamfer: {}, Part Accuracy: {}, Part Connect: {}".format(best_cd, best_acc,
                                                                                            best_conn) + "\n" + \
                       "==========================================================\n"
            log_writer.write(res_best + "\n")
            log_writer.flush()

    if args.rank == 0:
        summary_writer.close()
        log_writer.close()



if __name__ == '__main__':

    main()
