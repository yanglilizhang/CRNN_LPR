import torch.optim as optim
import time
from pathlib import Path
import os
import torch


def get_optimizer(config, model):
    optimizer = None

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def create_log_folder(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET  # 数据集名称  '360CC'
    model = cfg.MODEL.NAME  # 模型名称  'crnn'

    time_str = time.strftime('%Y-%m-%d-%H-%M')  # 时间 2023-12-14-16-09
    checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'  # 输出文件路径

    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'  # tensotborad日志路径
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    # encode的作用就是利用{'#': 0, '京': 1, ......}中的一一对应关系 将车牌名字转化为对应的数字
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    # plateDict2={'京':0, '京':1, '沪':2, ......, '-': 77}
    # 这个部分的作用首先是将预测输出的torch.size([128, 21])中的所有21个对应的最大概率的索引(下标为0 - 77)
    # 转化为对应的plateDict2中的字符, 这里为什么不是{‘-’: 0, …}即blank作为0, 这个原因可以见decode代码解析
    # 然后将转化好的21个字符中去掉重复的字符, 直接举个例子吧, -代表是blank, 以这个间隔, 删掉重复的字母
    # 左边是转化好的21个字符, 右边是得到的字符
    # 苏 - --E - -D - -3 - S - 11 - 22 - - = > 苏ED3S12, gt: 苏ED3S12
    # 赣 - -EE - -K - -1 - -3 - 22 - 0 - - = > 赣EK1320, gt: 赣EK1320
    # 渝 - AA - --66 - P - -8 - -6 - 11 - = > 渝A6P861, gt: 渝A6P861
    # 新 - -LL - -66 - 0 - 11 - 77 - 99 - = > 新L60179, gt: 新L60179
    # 鄂 - --N - -66 - MM - Y - 11 - 22 - = > 鄂N6MY12, gt: 鄂N6MY12
    # 豫 - --B - -D - -8 - -2 - 11 - 11 - = > 豫BD8211, gt: 渝BD8211
    # 苏 - -E - --1 - -1 - LL - 33 - LL - = > 苏E11L3L, gt: 苏E11L3L
    # 赣 - --C - -55 - 55 - T - -6 - 22 - = > 赣C55T62, gt: 赣C55T62
    # 川 - AA - --88 - 8 - -Y - -3 - 55 - = > 川A88Y35, gt: 川A88Y35
    # 例如 川 - AA - --88 = > 就会删掉一个A ，删掉所有的 -, 删掉一个8， 得到川A8

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        # 这个就是一个[21]的车牌字符的转化: 川-AA---88-8--Y--3-55- => 川A88Y35
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    # t[i] != 0 代表的是不为'-', not(i>0 and t[i-1]==t[i])表示的是当是第一个字符或者前后两个字符是相同的时候为True
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])   #将对应的索引转化为车牌字符
                return ''.join(char_list)
        # 这个是将[128, 21]分开为128个21, 即每一个车牌单独的送入上面的if length.numel() == 1:中
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def get_char_dict(path):
    with open(path, 'rb') as file:
        char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))
