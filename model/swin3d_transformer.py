import torch
import torch.nn as nn
from torch_points3d.modules.KPConv.kernels import KPConvLayer
from torch_scatter import scatter_softmax
from timm.models.layers import DropPath, trunc_normal_
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_geometric.nn import voxel_grid
from lib.pointops2.functions import pointops
from cd.chamfer import chamfer_distance






class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        mlp = list()
        for l in range(num_layers):
            if l == 0:
                mlp.append(nn.Linear(input_dim, hidden_dim))
                mlp.append(nn.LayerNorm(hidden_dim))
            elif l == num_layers - 1:
                mlp.append(nn.Linear(hidden_dim, output_dim))
                mlp.append(nn.LayerNorm(output_dim))
            else:
                mlp.append(nn.Linear(hidden_dim, hidden_dim))
                mlp.append(nn.LayerNorm(hidden_dim))
            mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.ModuleList(mlp)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


def grid_sample(pos, batch, size, start, end=None, return_p2v=True):





    cluster = voxel_grid(pos, batch, size, start=start,end=end)

    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)


    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k)
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1)
    p2v_map[mask] = torch.argsort(cluster)

    return cluster, p2v_map, counts

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.norm = norm_layer(in_channels) if norm_layer else None
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, feats, xyz, offset):

        n_offset, count = [int(offset[0].item()*self.ratio)+1], int(offset[0].item()*self.ratio)+1
        for i in range(1, offset.shape[0]):
            count += ((offset[i].item() - offset[i-1].item())*self.ratio) + 1
            n_offset.append(count)
        n_offset = torch.cuda.IntTensor(n_offset)
        idx = pointops.furthestsampling(xyz, offset, n_offset)
        n_xyz = xyz[idx.long(), :]

        feats = pointops.queryandgroup(self.k, xyz, n_xyz, feats, None, offset, n_offset, use_xyz=False)
        m, k, c = feats.shape
        feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)
        
        return feats, n_xyz, n_offset


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, quant_size, rel_query=True, rel_key=False, rel_value=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.quant_size = quant_size
        self.rel_query = rel_query
        self.rel_key = rel_key
        self.rel_value = rel_value

        quant_grid_length = int(window_size / quant_size)
        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_query_table, std=.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_key_table, std=.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_value_table, std=.02)

        self.quant_grid_length = quant_grid_length

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.softmax = nn.Softmax(dim=-1)


        self.linear1 = nn.Linear(3, 64)
        self.activate1 = nn.ReLU()
        self.linear2 = nn.Linear(64, num_heads)
        self.activate2 = nn.Softmax(dim=-1)

    def map_func(self, relative_position):
        return relative_position + self.quant_grid_length - 1

    def forward(self, feats, xyz, index_0, index_0_offsets, n_max, index_1, shift_size):
        """ Forward function.

        Args:
            feats: N, C
            xyz: N, 3
            p2v_idx: n, k
            counts: n, 
        """

        N, C = feats.shape
        

        qkv = self.qkv(feats).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        
        attn_flat = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max)

        xyz_quant = (xyz - xyz.min(0)[0] + shift_size) % self.window_size


        xyz_quant = xyz_quant // self.quant_size
        relative_position = xyz_quant[index_0] - xyz_quant[index_1]
        relative_position_index = self.map_func(relative_position)
        
        if self.rel_query and self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx_v3(query.float(), index_0_offsets.int(), n_max, key.float(), index_1.int(), self.relative_pos_query_table.float(), self.relative_pos_key_table.float(), relative_position_index.int())
        elif self.rel_query:
            relative_position_bias = pointops.dot_prod_with_idx(query.float(), index_0.int(), self.relative_pos_query_table.float(), relative_position_index.int())
        elif self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx(key.float(), index_1.int(), self.relative_pos_key_table.float(), relative_position_index.int())
        else:

            relative_position_bias = self.linear2(self.activate1(self.linear1(relative_position_index)))
            
        attn_flat = attn_flat + relative_position_bias
        
        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0)

        if self.rel_value:
            x = pointops.attention_step2_with_rel_pos_value_v2(softmax_attn_flat.float(), value.float(), index_0_offsets.int(), n_max, index_1.int(), self.relative_pos_value_table.float(), relative_position_index.int())
        else:
            x = pointops.attention_step2(softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int())
        x = x.view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, quant_size, 
            rel_query=True, rel_key=False, rel_value=False, drop_path=0.0,\
            mlp_ratio=4.0, qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=4):
        super().__init__()
        self.window_size = window_size
        self.mode = mode
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, quant_size=quant_size, 
            rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, feats, xyz, index_0, index_0_offsets, n_max, index_1, shift_size):






        short_cut = feats

        feats = self.norm1(feats)
        feats = self.attn(feats, xyz, index_0, index_0_offsets, n_max, index_1, shift_size)
        
        feats = short_cut + self.drop_path(feats)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))

        return feats

class BasicLayer(nn.Module):
    def __init__(self, depth, channel, num_heads, window_size, grid_size, quant_size, 
            rel_query=True, rel_key=False, rel_value=False, drop_path=0.0, mlp_ratio=4.0, qkv_bias=True, \
            qk_scale=None, norm_layer=nn.LayerNorm, downsample=None, ratio=0.25, k=16, out_channels=None):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.grid_size = grid_size
        self.max_window_counts = 64

        self.blocks = nn.ModuleList([SwinTransformerBlock(channel, num_heads, window_size, quant_size, 
            rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,\
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer) for i in range(depth)])

        self.downsample = downsample(channel, out_channels, ratio, k) if downsample else None

    """
    只要配置中把train data shuffle改为false，batchsize=20就能验证
    p2v_map[0:5]
    tensor([[ 79,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [ 14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [175,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [ 61, 229,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0]], device='cuda:0')
    counts[0:10]:tensor([1, 1, 1, 1, 2, 1, 1, 1, 1, 1], device='cuda:0')
    index_0[0:9]:tensor([ 79,   1,  14, 175,  61,  61, 229, 229, 137], device='cuda:0')
    index_1[0:9]:tensor([ 79,   1,  14, 175,  61, 229,  61, 229, 137], device='cuda:0')
    可以看到61，229，在两个索引中1个是横排，1个是竖排
    """"""
    p2v_map[0:5]
    tensor([[ 79,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [ 14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [175,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0],
            [ 61, 229,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0]], device='cuda:0')
    counts[0:10]:tensor([1, 1, 1, 1, 2, 1, 1, 1, 1, 1], device='cuda:0')
    index_0[0:9]:tensor([ 79,   1,  14, 175,  61,  61, 229, 229, 137], device='cuda:0') 横排列，该组中count有多少，同一个值就连续重复几次
    index_1[0:9]:tensor([ 79,   1,  14, 175,  61, 229,  61, 229, 137], device='cuda:0')
    可以看到61，229，在两个索引中1个是横排，1个是竖排
    p2v_map[236] 是包含元素最多的组，有107个元素
    [1507, 1510, 1512, 1514, 1517, 1520, 1521, 1524, 1525, 1527, 1528, 1531,
        1532, 1535, 1536, 1539, 1541, 1543, 1550, 1553, 1555, 1556, 1558, 1560,
        1561, 1566, 1568, 1569, 1570, 1571, 1572, 1575, 1576, 1581, 1583, 1584,
        1592, 1593, 1595, 1598, 1601, 1602, 1605, 1606, 1610, 1612, 1613, 1614,
        1615, 1622, 1623, 1625, 1634, 1636, 1640, 1643, 1645, 1647, 1650, 1652,
        1653, 1657, 1659, 1665, 1667, 1669, 1670, 1671, 1673, 1674, 1676, 1677,
        1679, 1680, 1683, 1685, 1686, 1691, 1692, 1695, 1698, 1707, 1711, 1713,
        1716, 1719, 1720, 1724, 1725, 1726, 1730, 1731, 1733, 1734, 1736, 1737,
        1738, 1739, 1742, 1743, 1744, 1747, 1748, 1751, 1752, 1755, 1756],
    p2v_map.unsqueeze(-1).expand(-1, -1, k)[236][0],第一个元素是1507，在index_0就重复了107，
    mask_mat[:236].sum()=45280,所以index_0[45280]=1507，index_0[45280+106]=1507
    index_0, indices = torch.sort(index_0) # index_0[45382]=1507，index_0[45382+106]=1507
    index_1 = index_1[indices] #[M,] index_1
    index_1[45280:45386] 为下面的数据，可以看到index_1就是（index_0中的同一个元素会重复k次）index_0这个重复了k次的同组的元素标号
    temp=tensor([1507, 1510, 1512, 1514, 1517, 1520, 1521, 1524, 1525, 1527, 1528, 1531,
        1532, 1535, 1536, 1539, 1541, 1543, 1550, 1553, 1555, 1556, 1558, 1560,
        1561, 1566, 1568, 1569, 1570, 1571, 1572, 1575, 1576, 1581, 1583, 1584,
        1592, 1593, 1595, 1598, 1601, 1602, 1605, 1606, 1610, 1612, 1613, 1614,
        1615, 1622, 1623, 1625, 1634, 1636, 1640, 1643, 1645, 1647, 1650, 1652,
        1653, 1657, 1659, 1665, 1667, 1669, 1670, 1671, 1673, 1674, 1676, 1677,
        1679, 1680, 1683, 1685, 1686, 1691, 1692, 1695, 1698, 1707, 1711, 1713,
        1716, 1719, 1720, 1724, 1725, 1726, 1730, 1731, 1733, 1734, 1736, 1737,
        1738, 1739, 1742, 1743, 1744, 1747, 1748, 1751, 1752, 1755],
       device='cuda:0')
    同理index_0[45382+231]=1510-index_0[45382+231+106]=1510,为第二个元素，所以index_1[indices[45382+231]:indices[45382+231+106]]也等于上面temp矩阵
    """
    def forward(self, feats, xyz, offset):



        window_size = torch.tensor([self.window_size]*3).type_as(xyz).to(xyz.device)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()

        v2p_map, p2v_map, counts = grid_sample(xyz, batch, window_size, start=None)




        N, C = feats.shape
        n, k = p2v_map.shape
        mask = torch.arange(k).unsqueeze(0).cuda() < counts.unsqueeze(-1)
        mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2))
        index_0 = p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
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
        
        shift_size = 1/2*window_size
        shift_v2p_map, shift_p2v_map, shift_counts = grid_sample(xyz+shift_size, batch, window_size, start=xyz.min(0)[0])
        
        n, k = shift_p2v_map.shape
        mask = torch.arange(k).unsqueeze(0).cuda() < shift_counts.unsqueeze(-1)
        mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2))
        index_0_shift = shift_p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
        index_1_shift = shift_p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]
        M = index_0_shift.shape[0]
 

        index_0_shift, indices_shift = torch.sort(index_0_shift)
        index_1_shift = index_1_shift[indices_shift]
        index_0_counts_shift = index_0_shift.bincount()
        n_max_shift = index_0_counts_shift.max()
        index_0_offsets_shift = index_0_counts_shift.cumsum(dim=-1)
        index_0_offsets_shift = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets_shift], 0)
        
        assert index_0_shift.shape[0] == index_1_shift.shape[0]
        assert index_0_shift.shape[0] == (shift_counts ** 2).sum()


        for i, blk in enumerate(self.blocks):
            index_0_blk = index_0 if i % 2 == 0 else index_0_shift
            index_0_offsets_blk = index_0_offsets if i % 2 == 0 else index_0_offsets_shift
            n_max_blk = n_max if i % 2 == 0 else n_max_shift
            index_1_blk = index_1 if i % 2 == 0 else index_1_shift
            shift_size_blk = 0.0 if i % 2 == 0 else shift_size

            feats = blk(feats, xyz, index_0_blk, index_0_offsets_blk, n_max_blk, index_1_blk, shift_size_blk)

        if self.downsample:
            feats_down, xyz_down, offset_down = self.downsample(feats, xyz, offset)
        else:
            feats_down, xyz_down, offset_down = None, None, None
            
        return feats, xyz, offset, feats_down, xyz_down, offset_down


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))

    def forward(self, feats, xyz, support_xyz, offset, support_offset, support_feats=None):

        feats = self.linear1(support_feats) + pointops.interpolation(xyz, support_xyz, self.linear2(feats), offset, support_offset)
        return feats, support_xyz, support_offset

class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, feats, xyz, batch, neighbor_idx):





        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats


class KPConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(nn.Linear(in_channels, d_2, bias=False), FastBatchNorm1d(d_2, momentum=bn_momentum), activation)
        self.unary_2 = torch.nn.Sequential(nn.Linear(d_2, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum), activation)
        self.kpconv = KPConvLayer(d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum)
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):




        
        shortcut = feats
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats += shortcut
        return feats


class Swin(nn.Module):
    def __init__(self, depths, channels, num_heads, window_sizes, up_k, \
                 grid_sizes, quant_sizes, rel_query=True, rel_key=False, rel_value=False, drop_path_rate=0.2, \
                 num_layers=4, concat_xyz=False, num_classes=13, ratio=0.25, k=16, prev_grid_size=0.04, sigma=1.0,
                 stem_transformer=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if stem_transformer:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma)
            ])
            self.layer_start = 0
        else:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma),
                KPConvResBlock(channels[0], channels[0], prev_grid_size, sigma=sigma)
            ])
            self.downsample = TransitionDown(channels[0], channels[1], ratio / 2, k)
            self.layer_start = 1

        self.layers = nn.ModuleList([BasicLayer(depths[i], channels[i], num_heads[i], window_sizes[i], grid_sizes[i], \
                                                quant_sizes[i], rel_query=rel_query, rel_key=rel_key,
                                                rel_value=rel_value, \
                                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                                downsample=TransitionDown if i < num_layers - 1 else None, \
                                                ratio=ratio, k=k,
                                                out_channels=channels[i + 1] if i < num_layers - 1 else None) for i in
                                     range(self.layer_start, num_layers)])

        self.upsamples = nn.ModuleList(
            [Upsample(up_k, channels[i], channels[i - 1]) for i in range(num_layers - 1, 0, -1)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)



        self.init_weights()

    def forward(self, feats, xyz, offset, batch, neighbor_idx):

        feats_stack = []
        xyz_stack = []
        offset_stack = []

        for i, layer in enumerate(self.stem_layer):
            feats = layer(feats, xyz, batch, neighbor_idx)

        feats = feats.contiguous()

        if self.layer_start == 1:
            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)

            feats, xyz, offset = self.downsample(feats, xyz,
                                                 offset)





        for i, layer in enumerate(self.layers):
            feats, xyz, offset, feats_down, xyz_down, offset_down = layer(feats, xyz, offset)

            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)
            if layer.downsample:
                feats = feats_down
                xyz = xyz_down
                offset = offset_down





        out = []
        start_pos = 0
        for i in range(offset.shape[0]):
            part_feature = self.maxpool((feats[start_pos:offset[i]]).unsqueeze(0).transpose(1, 2))



            start_pos = offset[i]
            out.append(part_feature.squeeze(0).squeeze(-1))
        out_feature = torch.stack(out)





        loss_points = None





























        return out_feature, loss_points

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
