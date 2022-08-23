from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
import numpy as np
import einops


def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c", # 现将基于 fh,fw 窗口 大小图片拉成一维，共有 ghxgw 个
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1]) # 将维度变为 n (gh gw) (fh fw) c
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


# MFI
class GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nn.Module): # 将输入 channel -> channel/2 ，u计算grid/v计算block 再concat
    """Get gating weights for cross-gating MLP block."""
    def __init__(self,nIn:int,Nout:int,H_size:int=128,W_size:int=128,input_proj_factor:int=2,dropout_rate:float=0.0,use_bias:bool=True,train_size:int=512):
        super(GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid, self).__init__()
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        if train_size == 512:
            self.grid_size = [[8, 8], [4, 4], [2, 2]]
        else:
            self.grid_size = [[6, 6], [3, 3], [2, 2]]

        self.block_size = [[int(H_size / l[0]), int(W_size / l[1])] for l in self.grid_size]
        self.input_proj_factor = input_proj_factor # 控制将输入 映射到多维，达到扩大channel 的目的.
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_end = nn.Linear(self.IN,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP_1 = nn.Linear((self.grid_size[0][0]*self.grid_size[0][1]),(self.grid_size[0][0]*self.grid_size[0][1]),bias=use_bias)

        self.Linear_Block_MLP_1 = nn.Linear((self.block_size[0][0]*self.block_size[0][1]),(self.block_size[0][0]*self.block_size[0][1]),bias=use_bias)

        self.Linear_grid_MLP_2 = nn.Linear((self.grid_size[1][0] * self.grid_size[1][1]),
                                           (self.grid_size[1][0] * self.grid_size[1][1]), bias=use_bias)

        self.Linear_Block_MLP_2 = nn.Linear((self.block_size[1][0] * self.block_size[1][1]),
                                            (self.block_size[1][0] * self.block_size[1][1]), bias=use_bias)

        self.Linear_grid_MLP_3 = nn.Linear((self.grid_size[2][0] * self.grid_size[2][1]),
                                           (self.grid_size[2][0] * self.grid_size[2][1]), bias=use_bias)

        self.Linear_Block_MLP_3 = nn.Linear((self.block_size[2][0] * self.block_size[2][1]),
                                            (self.block_size[2][0] * self.block_size[2][1]), bias=use_bias)

    def forward(self, x): # 去掉原 deterministic drop 后 未加 mask。
        n, h, w,num_channels = x.shape
        # n x h x w x c
        # input projection
        x = self.LayerNorm(x.float()) # 没有 float 报错
        x = self.Gelu(x)

        # grid 和 block 的大小都根据 给定的 grid_size or block_size 自动匹配另一个大小。即 grid_size 给定，自动计算 block_size
        # Get grid MLP weights
        gh1, gw1 = self.grid_size[0]
        fh1, fw1 = h // gh1, w // gw1
        u1 = block_images_einops(x, patch_size=(fh1, fw1)) # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u1 = u1.permute(0,3,2,1)

        u1 = self.Linear_grid_MLP_1(u1)
        u1 = u1.permute(0,3,2,1)
        u1 = unblock_images_einops(u1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        fh1, fw1 = self.block_size[0]
        gh1, gw1 = h // fh1, w // fw1
        v1 = block_images_einops(u1, patch_size=(fh1, fw1))
        v1 = v1.permute(0, 1, 3, 2)
        v1 = self.Linear_Block_MLP_1(v1)
        v1 = v1.permute(0, 1, 3, 2)
        v1 = unblock_images_einops(v1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        gh2, gw2 = self.grid_size[1]
        fh2, fw2 = h // gh2, w // gw2
        u2 = block_images_einops(v1, patch_size=(fh2, fw2))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u2 = u2.permute(0, 3, 2, 1)

        u2 = self.Linear_grid_MLP_2(u2)
        u2 = u2.permute(0, 3, 2, 1)
        u2 = unblock_images_einops(u2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        fh2, fw2 = self.block_size[1]
        gh2, gw2 = h // fh2, w // fw2
        v2 = block_images_einops(u2, patch_size=(fh2, fw2))
        v2 = v2.permute(0, 1, 3, 2)
        v2 = self.Linear_Block_MLP_2(v2)
        v2 = v2.permute(0, 1, 3, 2)
        v2 = unblock_images_einops(v2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        gh3, gw3 = self.grid_size[2]
        fh3, fw3 = h // gh3, w // gw3
        u3 = block_images_einops(v2, patch_size=(fh3, fw3))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u3 = u3.permute(0, 3, 2, 1)

        u3 = self.Linear_grid_MLP_3(u3)
        u3 = u3.permute(0, 3, 2, 1)
        u3 = unblock_images_einops(u3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        fh3, fw3 = self.block_size[2]
        gh3, gw3 = h // fh3, w // fw3
        v3 = block_images_einops(u3, patch_size=(fh3, fw3))
        v3 = v3.permute(0, 1, 3, 2)
        v3 = self.Linear_Block_MLP_3(v3)
        v3 = v3.permute(0, 1, 3, 2)
        v3 = unblock_images_einops(v3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        x = self.Linear_end(v3)
        x = self.dropout(x)
        return x # 不改变维度。# n h w c


class conv_T_y_2_x(nn.Module):
    """ Unified y Dimensional to x """
    def __init__(self,y_nIn,x_nOut):
        super(conv_T_y_2_x, self).__init__()
        self.x_c = x_nOut
        self.y_c = y_nIn
        self.convT = nn.ConvTranspose2d(in_channels=self.y_c, out_channels=self.x_c, kernel_size=(3,3),
                                        stride=(2, 2))
    def forward(self,x,y):
        # 考虑通用性，先将维度变为一致，在采样到相同大小
        y = self.convT(y)
        _, _, h, w, = x.shape
        y = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        return y


#TODO CGB
class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self,x_in:int,y_in:int,out_features:int,patch_size:[int,int],block_size:[int,int],grid_size:[int,int],dropout_rate:float=0.0,input_proj_factor:int=2,upsample_y:bool=True,use_bias:bool=True, train_size:int=512):
        super(CrossGatingBlock, self).__init__()
        self.IN_x = x_in
        self.IN_y = y_in
        self._h = patch_size[0]
        self._w = patch_size[1]
        self.features = out_features
        self.block_size=block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.Conv1X1_y = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y = nn.Linear(self.features,self.features,bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)

        self.ConvT = conv_T_y_2_x(self.IN_y,self.IN_x)
        self.fun_gx = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

        self.fun_gy = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

    def forward(self, x, y):
    # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            # 将 y 的维度调整为与 x 相同大小
            y = self.ConvT(x,y) # nn.ConvTranspose 反卷积

        x = self.Conv1X1_x(x)
        y = self.Conv1X1_y(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n x h x w x c
        y = y.permute(0, 2, 3, 1)
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)

        gx = self.fun_gx(x)
        # n x h x w x c
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.Linear_y(y)
        y = self.Gelu_y(y)

        gy = self.fun_gy(y)

        y = y * gx
        y = self.Linear_y_end(y)
        y = self.dropout_y(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.Linear_y_end(x)
        x = self.dropout_x(x)
        x = x + y + shortcut_x  # get all aggregated signals # 注意此处的 x 融合了来自y 的信息。
        x = x.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        y = y.permute(0, 3, 1, 2)
        return x, y


# 3D 输入  b x c x t x h x w
""" Cross Gating Block for 3D """

def block_images_einops_3D(x, patch_size:[int,int,int]):
    """Image to patches."""

    batch,depth,height, width, channels = x.shape
    grid_depth = depth // patch_size[0]
    grid_height = height // patch_size[1]
    grid_width = width // patch_size[2]

    x = einops.rearrange(
        x, "n (gd fd) (gh fh) (gw fw) c -> n (gd gh gw) (fd fh fw) c",  # 现将基于 fh,fw 窗口 大小图片拉成一维，共有 ghxgw 个
        gd=grid_depth,gh=grid_height, gw=grid_width, fd=patch_size[0],fh=patch_size[1], fw=patch_size[2])  # 将维度变为 n (gh gw) (fh fw) c
    return x


def unblock_images_einops_3D(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gd gh gw) (fd fh fw) c -> n (gd fd) (gh fh) (gw fw) c",
        gd=grid_size[0],gh=grid_size[1],gw=grid_size[2],fd=patch_size[0], fh=patch_size[1], fw=patch_size[2])
    return x

class GetSpatialGatingWeights_3D(nn.Module):  # 将输入 channel -> channel/2 ，u计算grid/v计算block 再concat
    """Get 3D gating weights for 3D cross-gating MLP block."""

    def __init__(self, nIn: int, Nout: int, D_size:int=128, H_size: int = 128, W_size: int = 128, block_size: [int,int,int] = [2,2,2],
                 grid_size: [int,int,int] = [2,2,2], input_proj_factor: int = 2, dropout_rate: float = 0.0,
                 use_bias: bool = True):
        super(GetSpatialGatingWeights_3D, self).__init__()
        self.D = D_size
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        self.block_size = block_size
        self.grid_size = grid_size
        self.input_proj_factor = input_proj_factor  # 控制将输入 映射到多维，达到扩大channel 的目的.
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_head = nn.Linear(self.IN, self.IN * input_proj_factor, bias=self.use_bias)
        self.Linear_end = nn.Linear(self.IN * self.input_proj_factor, self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP = nn.Linear((self.grid_size[0] * self.grid_size[1]* self.grid_size[2]),
                                         (self.grid_size[0] * self.grid_size[1]* self.grid_size[2]), bias=self.use_bias)
        self.Linear_Block_MLP = nn.Linear((self.block_size[0] * self.block_size[1] * self.block_size[2]),
                                          (self.block_size[0] * self.block_size[1] * self.block_size[2]), bias=self.use_bias)

    def forward(self, x):  # 去掉原 deterministic drop 后 未加 mask。
        n, d, h, w, num_channels = x.shape
        # n x d x h x w x c
        # input projection
        x = self.LayerNorm(x.float())  # 没有 float 报错
        x = self.Linear_head(x)  # 2C
        x = self.Gelu(x)

        u, v = torch.chunk(x,2,dim=-1)
        # grid 和 block 的大小都根据 给定的 grid_size or block_size 自动匹配另一个大小。即 grid_size 给定，自动计算 block_size
        # Get grid MLP weights
        gd, gh, gw = self.grid_size
        fd, fh, fw = d // gd, h // gh, w // gw
        u = block_images_einops_3D(u, patch_size=(fd ,fh , fw))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u = u.permute(0, 3, 2, 1)

        u = self.Linear_grid_MLP(u)
        u = u.permute(0, 3, 2, 1)
        u = unblock_images_einops_3D(u, grid_size=(gd,gh, gw), patch_size=(fd,fh, fw))
        # Get Block MLP weights
        fd, fh, fw = self.block_size
        gd, gh, gw = d // fd, h // fh, w // fw
        v = block_images_einops_3D(v, patch_size=(fd, fh, fw))
        v = v.permute(0, 1, 3, 2)
        v = self.Linear_Block_MLP(v)
        v = v.permute(0, 1, 3, 2)
        v = unblock_images_einops_3D(v, grid_size=(gd , gh, gw), patch_size=(fd ,fh , fw))
        x = torch.cat((u, v), dim=-1)
        x = self.Linear_end(x)
        x = self.dropout(x)
        return x  # 不改变维度。# n h w c


class conv_T_y_2_x_3D(nn.Module):
    """ Unified y Dimensional to x """

    def __init__(self, y_nIn, x_nOut):
        super(conv_T_y_2_x_3D, self).__init__()
        self.x_c = x_nOut
        self.y_c = y_nIn

        self.convT = nn.ConvTranspose3d(in_channels=self.y_c, out_channels=self.x_c, kernel_size=(3, 3, 3), stride=(2, 2, 2))

    def forward(self,x,y):
        # 考虑通用性，先将维度变为一致，在采样到相同大小
        y = self.convT(y)
        _,_,d,h,w, = x.shape
        y = nn.Upsample(size=(d,h,w), mode='trilinear', align_corners=True)(y)
        return y


# TODO CGB
class CrossGatingBlock_3D(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, x_in: int, y_in:int,out_features: int,patch_size:[int,int,int], block_size:[int,int,int], grid_size:[int,int,int], dropout_rate: float = 0.0,
                 input_proj_factor: int = 2, upsample_y: bool = True, use_bias: bool = True):
        super(CrossGatingBlock_3D, self).__init__()
        self.IN_x = x_in
        self.IN_y = y_in
        self._d = patch_size[0]
        self._h = patch_size[1]
        self._w = patch_size[2]

        self.features = out_features
        self.block_size = block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv3d(self.IN_x, self.features, (1, 1,1))
        self.Conv1X1_y = nn.Conv3d(self.IN_x, self.features, (1, 1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features, self.features, bias=use_bias)
        self.Linear_y = nn.Linear(self.features, self.features, bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features, self.features, bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features, self.features, bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)

        self.ConvT = conv_T_y_2_x_3D(self.IN_y,self.IN_x)

        # d h w 就是 x 的 大小
        self.fun_gx = GetSpatialGatingWeights_3D(nIn=self.features,Nout=self.features,
            D_size=self._d,
            H_size=self._h,
            W_size=self._w,
            block_size=self.block_size, grid_size=self.grid_size,input_proj_factor=self.input_proj_factor,dropout_rate=self.dropout_rate,use_bias=self.use_bias,
        )

        self.fun_gy = GetSpatialGatingWeights_3D(nIn=self.features,Nout=self.features,D_size=self._d,H_size=self._d,W_size=self._d,block_size=self.block_size,
            grid_size=self.grid_size,dropout_rate=self.dropout_rate,use_bias=self.use_bias)

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            # 将 y 的维度调整为与 x 相同大小
            y = self.ConvT(x,y) # nn.ConvTranspose 反卷积

        x = self.Conv1X1_x(x)
        n, num_channels, d , h, w = x.shape
        y = self.Conv1X1_y(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 4, 1)  # n x d x h x w x c
        y = y.permute(0, 2, 3, 4, 1)
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)
        # __init__(self,nIn:int,Nout:int,H_size:int=128,W_size:int=128,block_size:[int,...]=[2,2],grid_size:[int,...]=[2,2],input_proj_factor:int=2,dropout_rate:float=0.0,use_bias:bool=True):

        gx = self.fun_gx(x)
        # n x h x w x c
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.Linear_y(y)
        y = self.Gelu_y(y)
        gy = self.fun_gy(y)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.Linear_y_end(y)
        y = self.dropout_y(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.Linear_x_end(x)
        x = self.dropout_x(x)
        x = x + y + shortcut_x  # get all aggregated signals # 注意此处的 x 融合了来自y 的信息。
        x = x.permute(0, 4, 1, 2, 3)  # n x d x h x w x c --> n x c x d x h x w
        y = y.permute(0, 4, 1, 2, 3)
        return x, y
if __name__ == '__main__':
## test 3D
    a_t_3D = torch.tensor(np.random.randn(4,16,16,16,8)).float() # n d h w c 中间件的输入
    fun_1 = GetSpatialGatingWeights_3D(nIn=8,Nout=16,D_size=16,H_size=16,W_size=16,block_size=[2,2,2],grid_size=[2,2,2],input_proj_factor=2)
    for name, param in fun_1.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    b = fun_1(a_t_3D)
    print("b.shape:",b.shape)


    image_0_3D = torch.tensor(np.random.randn(4,8,32,32,32)).float() # 整个 核心块的输入。
    image_1_3D = torch.tensor(np.random.randn(4,4,16,16,16)).float()
    fun_CGB = CrossGatingBlock_3D(x_in=8,y_in=4,out_features=16,patch_size=[32,32,32],block_size=[2,2,2],grid_size=[2,2,2],dropout_rate=0.0,input_proj_factor=2,upsample_y=True,use_bias=True)
    g_1,g_2 = fun_CGB(image_0_3D,image_1_3D)
    print("g_1.shape:",g_1.shape)
    print("g_2.shape:",g_2.shape)

## test 2D
    a_t = torch.tensor(np.random.randn(4, 16, 16, 8)).float()  # n h w c 中间件的输入
    # def __init__(self, nIn: int, Nout: int, H_size: int = 128, W_size: int = 128, block_size: [int, ...] = [2, 2],
    # grid_size: [int, ...] = [2, 2], input_proj_factor: int = 2, dropout_rate: float = 0.0,
    # use_bias: bool = True):
    fun_1 = GetSpatialGatingWeights_2D(nIn=8, Nout=16, H_size=16, W_size=16, block_size=[2, 2], grid_size=[2, 2],
                                       input_proj_factor=2)
    for name, param in fun_1.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    b = fun_1(a_t)
    print("b.shape:", b.shape)

    # def __init__(self,x_in:int,y_in:int,out_features:int,block_size:int,grid_size:int,dropout_rate:float=0.0,
    # input_proj_factor:int=2,upsample_y:bool=True,use_bias:bool=True):

    image_0 = torch.tensor(np.random.randn(4, 8, 32, 32)).float()  # 整个 核心块的输入。
    image_1 = torch.tensor(np.random.randn(4, 4, 16, 16)).float()
    fun_CGB = CrossGatingBlock(x_in=8, y_in=4,patch_size=[32,32],out_features=16, block_size=[2, 2], grid_size=[2, 2], dropout_rate=0.0,
                               input_proj_factor=2, upsample_y=True, use_bias=True)
    g_1, g_2 = fun_CGB(image_0, image_1)
    print("g_1.shape:", g_1.shape)
    print("g_2.shape:", g_2.shape)
