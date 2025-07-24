import torch
import torch.nn as nn


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.1),
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
class ConvSelector2(nn.Module):
    def __init__(self, effect_nums=19, mode='Train', deleteNum=3, input_dim=120):
        super(ConvSelector2, self).__init__()
        self.mode = mode
        self.deleteNum = deleteNum

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.C2 = Conv(64, 128)
        self.C3 = Conv(128, 256)
        self.C4 = Conv(256, 512)
        self.C5 = Conv(512, 256)
        self.C6 = Conv(256, 128)
        self.C7 = Conv(128, 64)
        self.C8 = Conv(64, 1)
        self.fc = nn.Linear(640, 1, bias=True)

    def forward(self, x):
        if self.mode == 'CutUsage':
            _x = x.to(torch.float32)
            R1 = self.C1(_x)
            R2 = self.C2(R1)
            R3 = self.C3(R2)
            R4 = self.C4(R3)
            R5 = self.C5(R4)
            R6 = self.C6(R5)
            R7 = self.C7(R6)
            R8 = self.C8(R7)
            return self.fc(R8.squeeze(0).squeeze(0).squeeze(0))
        else:
            _x = x.to(torch.float32)
            R1 = self.C1(_x)
            R2 = self.C2(R1)
            R3 = self.C3(R2)
            R4 = self.C4(R3)
            R5 = self.C5(R4)
            R6 = self.C6(R5)
            R7 = self.C7(R6)
            R8 = self.C8(R7)
            return self.fc(R8.squeeze(0).squeeze(0).squeeze(0))
class ConvSelector(nn.Module):
    """
        reference:
        - unet_model
            https://blog.csdn.net/qimo601/article/details/125834066
        - math
            https://blog.csdn.net/perke/article/details/117732680
        - detail
            https://boardmix.cn/app/editor/zkDNK5EoUVKI3gqV5ubq8Q
            https://blog.csdn.net/weixin_44109827/article/details/124542394

        arguments：
        - input_size: feature size

        - effect_nums: the number of the unknown reasons that cause the change of Schumann Resonance
        - mode: if the unet_model is training, the mode equals to 'Train', otherwise, it equals to 'CutUsage'
        - deleteNum: the number of node that you want to delete while the mode equals to 'CutUsage'
    """

    def __init__(self, effect_nums=19, mode='Train', deleteNum=3, input_dim=120, rank=1):
        super(ConvSelector, self).__init__()
        self.mode = mode
        self.deleteNum = deleteNum
        self.rank = rank

        self.oneDimCnn = nn.Sequential()
        self.fc = nn.Sequential()
        level = 32
        for i in range(int(640 / level)):
            if i == 0:
                self.oneDimCnn.add_module('layer1_{}'.format(i),
                                          nn.Conv1d(in_channels=1, out_channels=1,
                                                    kernel_size=3, stride=2)
                                          )
            else:
                self.oneDimCnn.add_module('layer1_{}'.format(i),
                                          nn.Conv1d(in_channels=1, out_channels=1,
                                                    kernel_size=1, stride=1)
                                          )
        self.oneDimCnn.add_module('layer2_{}',
                                  nn.LeakyReLU()
                                  )
        print()
        self.nn1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.nn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.nn3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.nn4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.nn5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.nnLeaky = nn.LeakyReLU()
        self.nnSELU = nn.SELU()
        self.nnSwish = nn.Hardswish()

        # if self.mode != 'CutUsage':
        self.fc1 = nn.Linear(640, 1, bias=False)
        self.fc2 = self.nnSELU
        self.fc3 = nn.Linear(16, 1, bias=True)

        self.fc.add_module(
            'layer_linear1',
            nn.Linear(640, 1, bias=False),
        )
        self.fc.add_module(
            'layer_linear2',
            nn.LeakyReLU()
        )

        # init initial params
        for m in self.oneDimCnn.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def _normalize(self, x):
        __max = -999
        __min = 999
        for i in range(x.__len__()):
            if x[i] > __max:
                __max = x[i]
            if x[i] < __min:
                __min = x[i]
        x = (x - __min) / __max
        return x

    def forward(self, x):
        if self.mode == 'CutUsage':
            o = self.nn1(x.to(torch.float32))
            o = self.nnLeaky(o)
            o = self.nn2(o)
            o = self.nnLeaky(o)
            o = self.nn3(o)
            o = self.nnLeaky(o)
            o = self.nn4(o)
            o = self.nnLeaky(o)
            o = self.nn5(o)
            o = self.nnLeaky(o)
            o = torch.squeeze(o)
            print("o.shape")
            print(o.shape)
            if self.rank == 1:
                filter1 = [0, 2]
                for i in range(16):
                    if i not in filter1:
                        o[i] = self._normalize(o[i])
                o[0] *= 0
                o[2] = self._normalize((o[2] + 0.0023) * 10)

            if self.rank == 2:
                filter1 = [5, 9, 11, 3]
                filter2 = [1, 2, 7, 14]
                for i in range(16):
                    if i not in filter1:
                        o[i] = self._normalize((o[i] + 0.00211) * 10)
                    if i in filter1:
                        o[i] *= 0
                    if i in filter2:
                        o[i] = self._normalize(abs(o[i] + 0.5))

            if self.rank == 3:
                filter1 = [0, 1, 2]
                for i in range(16):
                    if i not in filter1:
                        o[i] = self._normalize(o[i])
                o[0] *= 0
                o[1] = self._normalize(o[1])
                o[2] = self._normalize((o[2] + 0.0023) * 10)

            o = o.transpose(0, 1)
            o = self.fc3(o)
            return o.transpose(0, 1)[0]
        else:
            o = self.nn1(x.to(torch.float32))
            o = self.nnLeaky(o)
            o = self.nn2(o)
            o = self.nnLeaky(o)
            o = self.nn3(o)
            o = self.nnLeaky(o)
            o = self.nn4(o)
            o = self.nnLeaky(o)
            o = self.nn5(o)
            o = self.nnLeaky(o)
            o = self.fc3(o.transpose(0, 1))
            return o.transpose(0, 1)[0]
