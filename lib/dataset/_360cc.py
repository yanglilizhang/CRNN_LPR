from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import platform
from alphabets import plate_chr


def cv_imread(path):  # 读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


# 数据集的读取
class _360CC(data.Dataset):
    def __init__(self, config, input_w=168, input_h=48, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.input_w = input_w  # 输入图片的宽
        self.input_h = input_h  # 输入图片的高
        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        char_file = config.DATASET.CHAR_FILE
        # with open(char_file, 'rb') as file:
        #     char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
        # with open(char_file, 'r',encoding='utf-8') as file:
        #     char_dict = {num: char.strip()  for num, char in enumerate(file.readlines())}
        # I resaved char_std_5990.txt in utf-8 format, so no need decode gbk
        # char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
        char_dict = {num: char.strip() for num, char in enumerate(plate_chr)}
        char_dict[0] = "blank"  # 训练的字符字典plateDict中第一个代表的是空白, 这个跟CTCLoss有关, 可以看一看CTCLoss后就可以理解了
        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                c = c.strip(" \n")
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        # img = cv2.imread(os.path.join(self.root, img_name))
        img = cv_imread(os.path.join(self.root, img_name))
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w, _ = img.shape

        # img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (self.input_w, self.input_h))
        # img = np.reshape(img, (48, 168, 3))
        # img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img / 255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])  # [h, w, c] -> [c, h, w] 这里没有brg -> rgb 在end2end预测的时候也没有 所以是可以的

        return img, idx
