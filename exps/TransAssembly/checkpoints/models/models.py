
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):

    def __init__(self, feat_dim):
        super(Predictor, self).__init__()

        self.mlp = nn.Linear(feat_dim, 512)

        self.trans = nn.Linear(512, 3)

        self.quat = nn.Linear(512, 4)
        self.quat.bias.data.zero_()

    """
        Input: * x F    (* denotes any number of dimensions, used as B x P here)
        Output: * x 7   (* denotes any number of dimensions, used as B x P here)
    """

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))

        trans = torch.tanh(self.trans(feat))

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out, trans, quat


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


class PredictorBox(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):

            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = torch.cat([x[...,0:3], F.relu(x[...,3:])+0.01], dim=-1)
        return x


class MLPRelu(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, last_relu = False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.last_relu = last_relu
    def forward(self, x):
        if self.last_relu:
            for i, layer in enumerate(self.layers):

                x = F.relu(layer(x))
        else:
            for i, layer in enumerate(self.layers):

                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, hparams):
        super().__init__()

        self.attn = Attn(hparams)

        self.hparams = hparams
        self.dropout = nn.Dropout(p=hparams.dropout)
        self.layernorm_attn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([self.hparams.hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.hparams.filter_size, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hparams.filter_size, self.hparams.hidden_size, bias=True))

    def preprocess_(self, X):
        return X


    def forward(self, X, mask):
        X = self.preprocess_(X)
        y = self.attn(X, mask)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(self.preprocess_(X))
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X


class Attn(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.kd = self.hparams.total_key_depth or self.hparams.hidden_size
        self.vd = self.hparams.total_value_depth or self.hparams.hidden_size
        self.q_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hparams.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hparams.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hparams.hidden_size, bias=False)
        assert self.kd % self.hparams.num_heads == 0
        assert self.vd % self.hparams.num_heads == 0


    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def dot_product_attention_message(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)

        v_w = v *  weights.unsqueeze(4)
        result = v_w.sum(dim=3)

        return result

    def forward(self, X, mask=None):
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)

        q = q.view(q.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.hparams.num_heads, self.kd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.hparams.num_heads, self.vd // self.hparams.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.hparams.num_heads) ** (-0.5)
        if self.hparams.attn_type == "with_mask":
            bias = 1e9 * (mask-1)
            bias = bias.unsqueeze(1).repeat(1, self.hparams.num_heads, 1, 1)
            result = self.dot_product_attention(q, k, v, bias=bias)
        if self.hparams.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.hparams.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.hparams.block_length
            pad = (0, 0, 0, (-len) % self.hparams.block_length)
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3)
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3)
                tail_q = q[:,:,1:]
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
                result = torch.cat([first_output, tail_output], 2)
                result = result[:,:,:X.shape[1],:]
            else:
                result = first_output[:,:,:X.shape[1],:]

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        return result

'''
class AttnRelativePosition(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.kd = self.vd = dim
        self.num_heads = num_heads
        self.q_dense = nn.Linear(dim, self.kd, bias=False)
        self.k_dense = nn.Linear(dim, self.kd, bias=False)
        self.v_dense = nn.Linear(dim, self.vd, bias=False)
        self.q_rel_dense = nn.Linear(dim, self.kd, bias=False)
        self.k_rel_dense = nn.Linear(dim, self.kd, bias=False)
        self.v_rel_dense = nn.Linear(dim, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, dim, bias=False)
        assert self.kd % num_heads == 0
        assert self.vd % num_heads == 0
        # self.mlp3 = MLP3(self.hparams.hidden_size) # add by csl

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v, weights # tf.matmul(A,C)=np.dot(A,C)= A@C都属于叉乘



    def dot_product_attention_message(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        #weights_rep = weights.unsqueeze(4).repeat(1, 1, 1, 1, v.shape[-1])
        v_w = v *  weights.unsqueeze(4) # 和乘以上面是等价的
        result = v_w.sum(dim=3)

        return result

    def forward(self, X, rel_pos, mask=None):
        q = self.q_dense(X) # [8,1024,512]
        k = self.k_dense(X)
        v = self.v_dense(X)
        q_rel = self.q_rel_dense(rel_pos)  # [8,1024,512]
        k_vel = self.k_rel_dense(rel_pos)
        v_vel = self.v_rel_dense(rel_pos)

        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3]) #[8,8,1024,64]
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.num_heads) ** (-0.5) #*0.125

        # for rel_pos
        weight_box = []
        for i in range(20):
            q_part = q[:,:, i:i+1,:]
            k_part = k[:,:, i:i+1,:]

            q_rel_part = q_rel[:,i]
            k_rel_part = k_rel[:, i]

            q_rel_part = q_rel_part.view(q_rel_part.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])  # [8,8,1024,64]
            k_rel_part = k_rel_part.view(k_rel_part.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
            logits_query_pos = torch.einsum("...kd,...qd->...qk", q_rel_part, q_part)
            logits_key_pos = torch.einsum("...kd,...qd->...qk", v_rel_part, v_part)
            logits_rel_pos = logits_query_pos + logits_key_pos
            weight_box.append(logits_rel_pos)
        bias_box = weight_box.cat(weight_box, dim=1)

        bias_mask = 1e9 * (mask-1) # mask中1为有效，0为无效，这样无效的就是-1e9也就是负无穷
        bias_mask = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        bias = bias_mask + bias_box
        # 得到x对应的上下文特征
        result_feature, weights = self.dot_product_attention(q, k, v, bias=bias)
        # 得到相对位置的特征
        out_rel_pos = []
        for i in range(20):
            v_rel_part = v_rel[:, i]
            v_rel_part = v_rel_part.view(v_rel_part.shape[:-1] + (self.num_heads, self.vd // self.hparams.num_heads)).permute([0, 2, 1, 3])
            out = weights @ v_rel_part
        out_rel_pos.append(out)
        result_rel_pos = torch.cat(out_rel_pos,dim=1)

        result = result_feature + result_rel_pos

        # 返回结果
        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        return result
    '''

class AttnRelativePosition(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.kd = self.vd = dim
        self.num_heads = num_heads
        self.q_dense = nn.Linear(dim, self.kd, bias=True)
        self.k_dense = nn.Linear(dim, self.kd, bias=True)
        self.v_dense = nn.Linear(dim, self.vd, bias=True)



        self.q_rel_dense = nn.Sequential(nn.Linear(192, self.kd, bias=True),
                      nn.ReLU(),
                      nn.Linear(self.kd, self.kd, bias=True))
        self.k_rel_dense = nn.Sequential(nn.Linear(192, self.kd, bias=True),
                      nn.ReLU(),
                      nn.Linear(self.kd, self.kd, bias=True))
        self.v_rel_dense = nn.Sequential(nn.Linear(192, self.kd, bias=True),
                      nn.ReLU(),
                      nn.Linear(self.kd, self.kd, bias=True))

        self.output_dense = nn.Linear(self.vd, dim, bias=False)
        assert self.kd % num_heads == 0
        assert self.vd % num_heads == 0
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        return weights @ v, weights


    def forward(self, X, rel_pos, mask=None):
        X = X.permute(1, 0, 2)
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        q_rel = self.q_rel_dense(rel_pos)
        k_rel = self.k_rel_dense(rel_pos)
        v_rel = self.v_rel_dense(rel_pos)


        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])

        n_factor = (self.kd // self.num_heads) ** (-0.5)

        weight_box = []
        for i in range(20):
            q_part = q[:,:, i:i+1,:]*n_factor
            k_part = k[:,:, i:i+1,:]*n_factor

            q_rel_part = q_rel[:,i]
            k_rel_part = k_rel[:,i]

            q_rel_part = q_rel_part.view(q_rel_part.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
            k_rel_part = k_rel_part.view(k_rel_part.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
            logits_query_pos = torch.einsum("...kd,...qd->...qk", q_rel_part, q_part)

            logits_key_pos = torch.einsum("...kd,...qd->...qk", k_rel_part, k_part)


            logits_rel_pos = logits_query_pos + logits_key_pos
            weight_box.append(logits_rel_pos)
        bias_box = torch.cat(weight_box, dim=2)

        mask_h = (~mask).int().unsqueeze(2).repeat(1, 1, 20)
        mask_v = (~mask).int().unsqueeze(1).repeat(1, 20, 1)
        mask = mask_h*mask_v
        bias_mask = 1e9 * (mask-1)
        bias_mask = bias_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        bias = bias_mask + bias_box

        q *= n_factor
        result_feature, weights = self.dot_product_attention(q, k, v, bias=bias)

        out_rel_pos = []
        for i in range(20):
            v_rel_part = v_rel[:, i]
            v_rel_part = v_rel_part.view(v_rel_part.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
            out = weights[:,:,i:i+1] @ v_rel_part

            out_rel_pos.append(out)
        result_rel_pos = torch.cat(out_rel_pos,dim=2)

        result = result_feature + result_rel_pos


        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        result = result.permute(1, 0, 2)
        return result