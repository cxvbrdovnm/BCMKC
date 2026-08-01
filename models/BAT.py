import math
import torch
from torch import nn
from timm.models.layers import Mlp, DropPath

class CEABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)  # from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.adap_t = Bi_direct_adapter(dim)
        self.adap2_t = Bi_direct_adapter()

    def forward(self, x, xi, x_mask=None, xi_mask=None):

        xori = x
        x_attn, attn = self.attn(self.norm1(x), x_mask, True)
        x = x + self.drop_path(x_attn) + self.drop_path(
            self.adap_t(self.norm1(xi)))  #########-------------------------adapter

        xi_attn, i_attn = self.attn(self.norm1(xi), xi_mask, True)
        xi = xi + self.drop_path(xi_attn) + self.drop_path(
            self.adap_t(self.norm1(xori)))  #########-------------------------adapter


        xori = x
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(
            self.adap2_t(self.norm2(xi)))  ###-------adapter

        xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(
            self.adap2_t(self.norm2(xori)))  ###-------adapter

        return x, xi


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x

class Bi_direct_adapter(nn.Module):
    def __init__(self, input_dim=768, dim=8):
        super().__init__()

        self.adapter_down = nn.Linear(input_dim, dim)
        self.adapter_up = nn.Linear(dim, input_dim)
        self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return x_up


if __name__ == '__main__':
    #init module
    hidden_dim = 768
    my_bat = CEABlock(dim=hidden_dim, num_heads=8)
    x_linear = nn.Linear(1400, hidden_dim)
    y_linear = nn.Linear(1400, hidden_dim)

    #get data
    x = torch.randn(30, 1400)  # b c
    y = torch.randn(30, 1400)  # b c

    #dat
    x = x_linear(x).unsqueeze(0)  # 1 b c
    y = y_linear(y).unsqueeze(0)  # 1 b c
    x_bat, y_bat = my_bat(x, y)
    print(x_bat.shape, y_bat.shape)