import shutil

import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
from alphabets import plate_chr


def cv_imread(path):  # 防止读取中文路径失败
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


# plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差
plate_color_list = ['黑色', '蓝色', '绿色', '白色', '黄色']


def decodePlate(preds):
    """
    对预测结果进行后处理，将数字编码转换为车牌字符。

    参数:
    preds: 预测结果的列表，每个元素代表一个字符的预测编码。

    返回值:
    plate: 车牌号码字符串，由预测结果转换得到。
    """

    pre = 0  # 上一个预测字符的编码
    newPreds = []  # 存储非重复的预测编码
    for i in range(len(preds)):
        # 仅当预测编码不为0且与上一个编码不同时，才添加到新列表中
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]  # 更新上一个预测字符的编码

    plate = ""  # 初始化车牌字符串
    # 将预测编码转换为对应的字符，添加到车牌字符串中
    for i in newPreds:
        plate += plate_chr[int(i)]

    return plate


def rec_pre_precessing(img, size=(48, 168)):  # 识别前处理
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = (img / 255 - mean_value) / std_value  # 归一化 减均值 除标准差
    img = img.transpose(2, 0, 1)  # h,w,c 转为 c,h,w
    img = img.reshape(1, *img.shape)  # channel,height,width转为batch,channel,height,channel
    return img


def get_plate_result(img, session_rec):
    """
    通过给定的图像和会话记录来识别车牌并返回结果。

    参数:
    img: 输入的图像，用于车牌识别。
    session_rec: ONNX会话记录，用于运行模型预测。

    返回值:
    plate_no: 识别出的车牌号码。
    plate_color: 车牌的颜色。
    """
    # 图像的预处理
    img = rec_pre_precessing(img)
    # 运行ONNX模型，分别获取车牌字符和颜色的预测结果
    y_onnx_plate, y_onnx_color = session_rec.run(
        [session_rec.get_outputs()[0].name, session_rec.get_outputs()[1].name],
        {session_rec.get_inputs()[0].name: img}
    )
    # 从预测结果中找出最有可能的车牌字符和颜色
    index = np.argmax(y_onnx_plate, axis=-1)
    index_color = np.argmax(y_onnx_color)
    plate_color = plate_color_list[index_color]
    # print(y_onnx[0])
    # 解码得到车牌号码
    plate_no = decodePlate(index[0])
    return plate_no, plate_color


def allFilePath(rootPath, allFIleList):
    """
    遍历给定根路径下的所有文件和文件夹，并将所有文件的路径添加到列表中。

    :param rootPath: str, 根路径的字符串表示。
    :param allFIleList: list, 存储所有文件路径的列表。
    :return: None
    """
    fileList = os.listdir(rootPath)  # 获取根路径下的所有文件和文件夹列表
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            # 如果是文件，则将其路径添加到列表中
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            # 如果是文件夹，则递归遍历该文件夹
            allFilePath(os.path.join(rootPath, temp), allFIleList)


# 识别模型（包含颜色）的推理
# python onnx_infer_merge.py --onnx_file saved_model/plate_rec_color.onnx  --image_path images/test.jpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', type=str, default='saved_model/plate_rec_color_anpr.onnx',
                        help='model.pt path(s)')  # 识别模型
    # parser.add_argument('--image_path', type=str, default='images', help='source')
    # parser.add_argument('--image_path', type=str, default='/Users//Downloads/CCPDandCRPD车牌识别训练集/CCPD_CRPD_OTHER_ALL', help='source')
    # parser.add_argument('--image_path', type=str, default='/Volumes/Samsung USB/202311-copy', help='source')
    parser.add_argument('--image_path', type=str, default=r'/Volumes/Samsung USB/small/202312small3', help='source')
    parser.add_argument('--img_h', type=int, default=48, help='inference size (pixels)')
    parser.add_argument('--img_w', type=int, default=168, help='inference size (pixels)')
    # 主程序块：用于处理输入的命令行参数，加载ONNX模型，对单个图像或图像列表进行处理，并输出结果。
    opt = parser.parse_args()  # 解析命令行参数
    providers = ['CPUExecutionProvider']  # 指定执行提供者为CPU
    session_rec = onnxruntime.InferenceSession(opt.onnx_file, providers=providers)  # 加载ONNX模型

    # 检查是否是单个文件输入
    if os.path.isfile(opt.image_path):
        img = cv_imread(opt.image_path)  # 读取图像
        # 处理图像通道，如果是BGRA则转换为BGR
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        plate, plate_color = get_plate_result(img, session_rec)  # 识别车牌并获取颜色
        print(f"{plate} {plate_color} {opt.image_path}")  # 输出结果
    else:
        file_list = []  # 用于存储文件路径的列表
        right = 0  # 记录正确识别的车牌数量
        # 如果是目录输入，则遍历目录下所有文件
        allFilePath(opt.image_path, file_list)
        # print(f"file_list:{file_list}")
        # 创建一个新的目录来存放识别错误的车牌图像

        error_dir = "error_plates"
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        for pic_ in file_list:
            img = cv_imread(pic_)  # 读取图像
            # 同样处理图像通道
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            plate, plate_color = get_plate_result(img, session_rec)  # 识别车牌
            plate_ori = pic_.split(os.sep)[-1].split('_')[0]  # 从文件名中提取原始车牌号
            # print(pic_,plate_color)
            if (plate == plate_ori):  # 比较识别结果与原始车牌号
                right += 1
            else:
                # 打印识别错误的信息
                print(plate_ori, "rec error info: ", plate, pic_, plate_color)
                # 将识别错误的车牌图像移动到错误目录
                shutil.move(pic_, os.path.join(error_dir, plate_ori))
                # print(plate,pic_name)
            print(f"{plate} {plate_color} {pic_}")  # 输出结果
        print("sum:%d ,right:%d , accuracy: %f" % (len(file_list), right, right / len(file_list)))
