# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.last_feature_size=dims[-1]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        # return x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_list(self, x):
        features=[]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features # 返回不同阶段的feature


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_mlp(input_size, hidden_size, output_size, num_hidden_layers):
    layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_size, output_size))
    model = nn.Sequential(*layers)
    return model

class FusionMoe(nn.Module):
    def __init__(self, vit_features_dim, cnn_features_dim, num_experts, output_dim, mlp_layer=1):
        super(FusionMoe, self).__init__()
        hidden_size = 2048
        if mlp_layer == 1:
            self.expert_vit = nn.Linear(vit_features_dim, output_dim)
            self.expert_cnn = nn.Linear(cnn_features_dim, output_dim)
            self.gate = nn.Linear(vit_features_dim + cnn_features_dim, num_experts)
        else:
            self.expert_vit=create_mlp(vit_features_dim,hidden_size,output_dim, mlp_layer)
            self.expert_cnn=create_mlp(cnn_features_dim,hidden_size,output_dim, mlp_layer)
            self.gate=create_mlp(vit_features_dim + cnn_features_dim,hidden_size,num_experts, mlp_layer)
        # 门控网络，根据输入特征决定每个专家的权重
        # self.gate = nn.Linear(vit_features_dim + cnn_features_dim, num_experts)
        self.num_experts = num_experts
        self.output_dim = output_dim

    def forward(self, vit_features, cnn_features):
        #扩展cnn_feature
        bs,spatial_size,_= vit_features.size()
        cnn_features_expand = cnn_features.unsqueeze(1).expand(-1,spatial_size,-1)
        # 专家处理
        expert_out_vit = self.expert_vit(vit_features)
        expert_out_cnn = self.expert_cnn(cnn_features_expand)
        # 门控网络输出权重
        combined_features = torch.cat([vit_features, cnn_features_expand], dim=-1)
        gate_scores = self.gate(combined_features)
        gate_weights = F.softmax(gate_scores, dim=-1)
        # 将每个专家的输出和权重结合起来
        expert_outputs = torch.stack([expert_out_vit, expert_out_cnn], dim=1)  # shape: [batch_size, num_experts, output_dim]
        fused_output = torch.einsum('bisk,bsi->bsk', expert_outputs, gate_weights)  # b: batch, i: expert index, k: output dimension
        return fused_output

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)

def create_covnet(model_name,drop_path_rate=0.,precision="fp16"):
    if model_name=='covnextv2_large':
        model = convnextv2_large(drop_path_rate=drop_path_rate)
        local_file = '/home/iv/Intern_new/ChenBin/outpainting/LAVIS/ConvNeXt-V2/convnextv2_large_22k_224_ema.pt'
        # state_dict = torch.load(cached_file, map_location="cpu")
        state_dict = torch.load(local_file, map_location="cpu")
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print('already load covnet pretrained weight!')
        #     print(incompatible_keys)

    elif model_name=='covnextv2_huge':
        model = convnextv2_huge(drop_path_rate=drop_path_rate)
        local_file = '/home/iv/Intern_new/ChenBin/outpainting/LAVIS/ConvNeXt-V2/convnextv2_huge_22k_384_ema.pt'
        # state_dict = torch.load(cached_file, map_location="cpu")
        state_dict = torch.load(local_file, map_location="cpu")
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print('already load covnet pretrained weight!')
        #     print(incompatible_keys)
    else:
        raise ValueError(f"{model_name} is not implement !")

    if precision=='fp16':
        convert_weights_to_fp16(model)

    return model

if __name__ == '__main__':
    zz=create_mlp(10,2048,3,2)
    print(1)