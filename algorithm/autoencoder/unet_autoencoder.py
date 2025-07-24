"""

"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cmath


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合            # nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)
# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)

# UNet网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 下采样部分
        # R1 = self.C1(x)
        # R2 = self.C2(self.D1(R1))
        # R3 = self.C3(self.D2(R2))
        #
        # # 上采样部分
        # # 上采样的时候需要拼接起来
        # O3 = self.C8(self.U3(R3, R2))
        # O4 = self.C9(self.U4(O3, R1))

        # 输出，这里大小跟输入是一致的
        return self.pred(O4)

# 非线性回归器(多通道的2D卷积过程)
class NonlinearRegressor(nn.Module):

    def __init__(self):
        super(NonlinearRegressor, self).__init__()
        self.C1 = Conv(3, 1)
        self.C2 = Conv(1, 1)
        self.C3 = Conv(1, 1)
        self.layer = nn.Sequential(
            self.C1,
            self.C2,
            self.C3,
        )
    def forward(self, x):
        return self.layer(x)
def normalize(image):
    _max = 0
    _min = 999
    for i in range(0, image.__len__()):
        for j in range(0, image[0].__len__()):
            if _max < image[i][j]:
                _max = image[i][j]
            if _min > image[i][j]:
                _min = image[i][j]
    return (image - _min) / _max, _max, _min
def matrixPow(matrix, n):
    if type(matrix) == list:
        matrix = np.array(matrix)
    if n == 1:
        return matrix
    if n % 2 == 0:
        # 偶数则返回原矩阵
        return matrix
    else:
        # 奇数次幂
        y = matrix
        for i in range(1, n):
            if i % 2 == 0:
                y = torch.mm(y, matrix) / 640
            else:
                y = torch.mm(y, matrix.transpose(0, 1)) / 640
        # _, _max, _min = normalize(y.detach().numpy())
        return y
# 主模块
class CombinedUNet(nn.Module):

    def __init__(self):
        super(CombinedUNet, self).__init__()
        self.UNet = UNet()
        self.NonlinearRegressor = NonlinearRegressor()

    def forward(self, x):

        _ = self.UNet(x.unsqueeze(0).unsqueeze(0)).squeeze(0)
        xi = _.detach().numpy()
        _size = _.size(0)

        # _y = torch.zeros(640, 640)
        _y = torch.zeros(560, 560)

        sub_y = []
        sinx = []
        cosx = []
        ex = []
        dYs = []
        for index in range(0, _size):
            '''
            '''
            # sub_y.append(self.calXi(_[index]))
            # ex.append(_[index].exp())
            # xEx.append(torch.mul(_[index], ex[index]))
            # _y = torch.add(_y, sub_y[index])
            sinx.append(_[index].sin())
            cosx.append(_[index].cos())
            ex.append(_[index].exp())
            fx = torch.mul((-sinx[index] + 1), ex[index])
            sub_y.append(fx)
            _y = torch.add(_y, sub_y[index])
        '''
            需要返回的有一阶导、二阶导
            dYs = [Y, Y', Y''] 
        '''
        for i in range(0, sub_y.__len__()):
            dYs.append([
                sub_y,
                torch.mul((-sinx[i] - cosx[i] + 1), ex[index]),
                torch.mul((-2*cosx[i] + 1), ex[index]),
            ])
        return _y, xi, dYs


if __name__ == '__main__':
    a = torch.zeros(2, 3, 256, 256)
    b = a + 1
    print(3-b)
