import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import CNNBlockBase, Conv2d
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AddFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)

        input_proj = []
        # backbone_dim = [192, 384, 768, 768]
        backbone_dim = [256, 512, 1024, 1024]
        for l in range(4):
            input_proj_l = nn.Sequential(
                LayerNorm(backbone_dim[l]),
                Conv2d(
                    backbone_dim[l],
                    backbone_dim[l],
                    kernel_size=1,
                )
            )
            input_proj.append(input_proj_l)
            weight_init.c2_xavier_fill(input_proj_l[-1])
        self.input_proj = nn.ModuleList(input_proj)

        self.text_proj = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        weight_init.c2_xavier_fill(self.text_proj[-1])

    def forward(self, idx, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple, image=True):

        if image:
            # x: [N,L,C] y: [N,C,H,W]
            # print(x.shape, 'x imag')
            # print(y.shape, 'y imag')
            y = (
                F.interpolate(
                    self.input_proj[idx](y.contiguous()),
                    size=spatial_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(x.shape)
            )
            x = x + y

        else:
            # x: [b,L,C] y: [1,bt,C]
            # print(x.shape, 'x text') #1 12 256
            # print(y.shape, 'y text') #1 5 256
            b, l, c = x.shape
            y = y.mean(dim=1, keepdim=True) #1 b c
            y = y.repeat(l, 1, 1) #l b c
            y = self.text_proj(y.permute(1, 2, 0).contiguous()) #bt c l
            y = y.permute(0, 2, 1)
            x = x + y

            #print(x.shape, 'text x') #bt l c
        return x

class AddFusion_2(CNNBlockBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C] y: [N,C,H,W]
        y = (
            F.interpolate(
                self.input_proj(y.contiguous()),
                size=spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(x.shape)
        )
        x = x + y
        return x


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int):
    if fusion_type == "add":
        return AddFusion(in_channels, out_channels)
    else:
        raise ValueError("Unknown fusion type: {}".format(fusion_type))
