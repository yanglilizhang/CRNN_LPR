import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, trt=False):
        """
        初始化myNet_ocr网络模型。

        参数:
        - cfg: 网络配置列表，用于定义卷积层的结构。如果为None，则使用默认配置。
        - num_classes: 类别的数量，用于定义分类器的输出维度。表示有78个需要识别的字符
        - export: 一个布尔值，标识是否为导出模型。
        - trt: 一个布尔值，标识是否使用TensorRT优化模型。

        返回值:
        - 无
        """
        super(myNet_ocr, self).__init__()
        if cfg is None:
            # 设置默认的网络配置（大模型）
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
        self.feature = self.make_layers(cfg, True)  # 构建卷积层特征提取部分
        self.export = export  # 导出模型标志
        self.trt = trt  # TensorRT优化模型标志
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)  # 定义位置编码的池化层
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)  # 定义新的卷积层，用于类别预测
        # 下面的注释掉的代码块可能是用于实验不同池化层效果的遗留代码
        # self.classifier = nn.Linear(cfg[-1], num_classes)  # 用于原始分类任务的全连接层 (被注释)
        # self.loc = nn.MaxPool2d((2, 2), (5, 1), (0, 1), ceil_mode=True)  # 另一种位置编码的池化层方式 (被注释)
        # self.loc = nn.AvgPool2d((2, 2), (5, 2), (0, 1), ceil_mode=False)  # 再一种位置编码的池化层方式 (被注释)
        # self.newBn = nn.BatchNorm2d(num_classes)  # 可能用于批量归一化的层 (被注释)

    def make_layers(self, cfg, batch_norm=False):
        """
        构建网络层列表。

        参数:
        - cfg: 一个配置列表，用于指定每一层的类型和通道数。其中，数字代表卷积层的输出通道数，'M'代表最大池化层。
        - batch_norm: 一个布尔值，指定是否在卷积层后添加批量标准化层。默认为False。

        返回值:
        - 返回一个包含多个层的nn.Sequential模型，可以被用作PyTorch模型的一部分。
        """
        layers = []  # 初始化层列表
        in_channels = 3  # 输入通道数，默认为RGB图像的3通道

        for i in range(len(cfg)):
            if i == 0:
                # 第一层卷积，使用较大的卷积核
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]  # 更新输入通道数
            else:
                # 非第一层，可能是卷积层或池化层
                if cfg[i] == 'M':
                    # 如果配置为'M'，则添加最大池化层
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    # 添加卷积层，并根据配置决定是否添加批量标准化层
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        实现前向传播过程。

        参数:
        - x输入张量，期望的输入形状取决于模型的初始化和特征提取需求。

        返回值:
        - 如果self.export为True，则返回经过处理的张量conv，用于导出模型。
        - 如果self.export为False，则返回经过log_softmax激活函数处理的张量output，用于预测。
        """
        # 特征提取
        x = self.feature(x)
        # 位置回归
        x = self.loc(x)
        # 新CNN层处理
        x = self.newCnn(x)
        # 注释掉的BN层（批量归一化）
        # x=self.newBn(x)

        if self.export:
            # 对张量进行维度调整，用于导出模型
            conv = x.squeeze(2)  # 从[b, c, h, w]调整为[b, c, w]
            conv = conv.transpose(2, 1)  # 从[b, c, w]调整为[w, b, c]
            # 如果启用了TRT（TensorRT），则对张量进行最大值激活，用于加速推理
            if self.trt:
                conv = conv.argmax(dim=2)
                conv = conv.float()
            print("--->conv.shape:", conv.shape)
            return conv
        else:
            # 获取张量的尺寸，用于后续处理
            b, c, h, w = x.size()
            # 断言卷积层的高度必须为1，这是模型的特定要求
            assert h == 1, "the height of conv must be 1"
            # 经过维度调整和排列变换，准备进行softmax激活
            conv = x.squeeze(2)  # 从[b, c, h, w]调整为[b, c, w]
            conv = conv.permute(2, 0, 1)  # 从[w, b, c]调整为[b, w, c]
            # 应用log_softmax激活函数，准备进行预测
            output = F.log_softmax(conv, dim=2)
            # 注释掉的softmax激活函数（可能用于验证或其他目的）
            # output = torch.softmax(conv, dim=2)
            print("--->output.shape:", output.shape)
            return output


if __name__ == '__main__':
    """
    初始化一个用于OCR（Optical Character Recognition，光学字符识别）的模型，并对其进行测试。

    参数:
    - x: 一个形状为(1, 3, 48, 168)的torch张量，代表输入的图像数据，其中1表示批次大小，3表示通道数（RGB），48和168分别表示图像的高度和宽度。
    - cfg: 一个列表，包含了模型的结构配置，如卷积核数量和'M'代表的池化层。
    - num_classes: 整数，表示模型要识别的类别数量。
    - export: 布尔值，用于指定模型是否以导出为目的进行构建。

    返回值:
    - 无。此函数不返回任何值，但会打印模型的输出形状。
    """
    x = torch.randn(1, 3, 48, 168)  # 初始化输入张量
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]  # 定义模型结构配置
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)  # 根据配置创建模型实例
    # print(model)  # 打印模型结构，注释掉了以符合要求
    out = model(x)  # 通过模型进行前向传播
    print(out.shape)  # 打印模型输出的形状 torch.Size([1, 21, 78])

