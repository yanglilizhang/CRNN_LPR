from plateNet import myNet_ocr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import argparse
from alphabets import plate_chr
from LPRNet import build_lprnet


# 通过给定的路径读取图片，特别适用于处理包含中文字符的路径。
def cv_imread(path):  # 读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    """
    遍历指定路径下的所有文件和子目录，将所有文件的路径添加到列表中。

    :param rootPath: 指定的根路径字符串
    :param allFIleList: 用于存储所有文件路径的列表，遍历过程中会向此列表添加文件路径
    """
    fileList = os.listdir(rootPath)  # 获取根路径下的所有文件和目录名列表
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            # 如果是文件，则将其路径添加到列表中
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            # 如果是目录，则递归调用此函数处理该目录下的所有文件和子目录
            allFilePath(os.path.join(rootPath, temp), allFIleList)


mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    """
    解码预测结果，去除连续的重复元素和0元素。

    参数:
    preds: 一个整数列表，表示预测的结果序列。

    返回值:
    一个新的整数列表，其中去除了输入列表中的连续重复元素和0元素。
    """
    pre = 0  # 初始化上一个元素为0
    newPreds = []  # 初始化一个新的列表存储解码后的结果

    # 遍历预测结果列表
    for i in range(len(preds)):
        # 如果当前元素不为0且与上一个元素不相同，则将其添加到新列表中
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]  # 更新上一个元素为当前元素

    return newPreds


def image_processing(img, device, img_size):
    """
    对图像进行预处理，包括调整图像大小、归一化、转换为张量以及移动到指定设备上。

    参数:
    img: 输入图像，可以是OpenCV格式的图像，要求是BGR颜色空间。
    device: 指定运行张量的设备，如'cpu'或'cuda:0'。
    img_size: 目标图像大小，以元组形式表示，如(224, 224)。

    返回值:
    调整大小、归一化并转换为张量后的图像，准备好在指定设备上进行模型输入。
    """
    # 调整图像大小
    img_h, img_w = img_size
    img = cv2.resize(img, (img_w, img_h))

    # 归一化图像
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value  # 使用预定义的均值和标准差进行归一化
    img = img.transpose([2, 0, 1])  # 调整通道顺序以符合深度学习模型的输入要求
    img = torch.from_numpy(img)  # 将numpy数组转换为torch张量

    # 将图像移动到指定的设备上，并调整其形状以适配模型输入
    img = img.to(device)
    img = img.view(1, *img.size())  # 调整张量形状，为模型输入做准备
    return img


def get_plate_result(img, device, model, img_size):
    """
    从图像中识别车牌并返回结果

    参数:
    img: 输入的图像，可以是图像路径或图像数组；
    device: 指定运行模型的设备，如'cpu'或'cuda';
    model: 用于识别车牌的模型；
    img_size: 模型期望的输入图像大小。

    返回值:
    plate: 识别出的车牌字符字符串。
    """
    # 对输入图像进行预处理，以便于模型输入
    # img = cv2.imread(image_path)
    input = image_processing(img, device, img_size)
    # 使用模型进行预测
    preds = model(input)
    # 获取预测结果中概率最高的类别
    preds = preds.argmax(dim=2)
    # print(preds)
    # 将模型预测结果转换为可读的字符
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    # 将预测的字符进行组合，形成最终的车牌号码
    plate = ""
    for i in newPreds:
        # plate_chr="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        plate += plate_chr[int(i)]
    return plate


def init_model(device, model_path):
    """
       初始化模型。

       参数:
       - device: 指定模型运行的设备，可以是'cpu'或'cuda:0'等。
       - model_path: 模型权重文件的路径。

       返回:
       - 初始化后的模型。
       """
    # 从指定路径加载模型权重和配置
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']

    # 根据配置创建模型实例
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)  # export  True 用来推理
    # model =build_lprnet(num_classes=len(plate_chr),export=True)

    # 加载模型状态
    model.load_state_dict(model_state, strict=False)
    # 将模型转移到指定的设备上
    model.to(device)
    # 设置模型为评估模式
    model.eval()
    return model


# 主要负责单图/批量识别及准确率计算-字符识别模型测试
# python demo.py --model_path saved_model/best.pth --image_path images/test.jpg or your/model/path
if __name__ == '__main__':
    # 解析命命令行参数
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='saved_model/best.pth', help='model.pt path(s)')
    parser.add_argument('--model_path', type=str, default='saved_model/plate_rec.pth', help='model.pt path(s)')
    # parser.add_argument('--image_path', type=str, default='images/test.jpg', help='source')
    parser.add_argument('--image_path', type=str, default='images', help='source')
    # parser.add_argument('--image_path', type=str, default='/mnt/Gu/trainData/plate/new_git_train/val_verify', help='source')
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w', type=int, default=168, help='width')
    parser.add_argument('--LPRNet', action='store_true', help='use LPRNet')  # True代表使用LPRNet ,False代表用plateNet
    parser.add_argument('--acc', type=bool, default='false', help=' get accuracy')  # 标记好的图片，计算准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device =torch.device("cpu")
    opt = parser.parse_args()
    img_size = (opt.img_h, opt.img_w)
    # 加载模型（方法内部需要手动选择需要的模型类型）
    model = init_model(device, opt.model_path)

    # 处理单张图片或目录
    if os.path.isfile(opt.image_path):  # 判断是单张图片还是目录
        # 单张图片识别
        right = 0
        begin = time.time()
        img = cv_imread(opt.image_path)
        # 图片格式转换（若需要）
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        plate = get_plate_result(img, device, model, img_size)
        print(plate)
    elif opt.acc:
        # 计算目录下图片的准确率
        file_list = []
        right = 0
        allFilePath(opt.image_path, file_list)
        for pic_ in file_list:
            try:
                pic_name = os.path.basename(pic_)
                img = cv_imread(pic_)
                if img.shape[-1] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                plate = get_plate_result(img, device, model, img_size)
                plate_ori = pic_.split('/')[-1].split('_')[0]
                # print(plate,"---",plate_ori)
                if (plate == plate_ori):
                    right += 1
                else:
                    print(plate_ori, "rec as ---> ", plate, pic_)
                    # print(plate,pic_name)
            except:
                print("error")
        print("sum:%d ,right:%d , accuracy: %f" % (len(file_list), right, right / len(file_list)))
    else:
        # 遍历目录下所有图片进行识别
        file_list = []
        allFilePath(opt.image_path, file_list)
        for pic_ in file_list:
            try:
                pic_name = os.path.basename(pic_)
                img = cv_imread(pic_)
                # 图片格式转换（若需要）
                if img.shape[-1] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                plate = get_plate_result(img, device, model)
                print(plate, pic_name)
            except:
                print("error")
