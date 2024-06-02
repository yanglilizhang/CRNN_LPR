# from plateNet import myNet_ocr
import shutil

from colorNet import myNet_ocr_color
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import argparse
from alphabets import plate_chr
from LPRNet import build_lprnet


def cv_imread(path):  # 读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


color = ['黑色', '蓝色', '绿色', '白色', '黄色']
mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds


def image_processing(img, device, img_size):
    img_h, img_w = img_size
    img = cv2.resize(img, (img_w, img_h))
    # img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, img_size):
    # img = cv2.imread(image_path)
    input = image_processing(img, device, img_size)
    preds, preds_color = model(input)
    preds = preds.argmax(dim=2)
    preds_color = preds_color.argmax()
    preds_color = preds_color.item()
    # print(preds)
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate, color[preds_color]


def init_model(device, model_path):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr_color(num_classes=len(plate_chr), export=True, cfg=cfg, color_num=5)  # export  True 用来推理
    # model =build_lprnet(num_classes=len(plate_chr),export=True)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


#  测试颜色分支训练出的模型
#  python demo_plate_color.py --model_path saved_model/plate_rec_color.pth --image_path images/test.jpg
#  python demo_plate_color.py --model_path saved_model/plate_rec_color.pth --image_path images
#  python demo_plate_color.py --model_path saved_model/plate_rec_color_backup.pth --image_path images
#  python demo_plate_color.py --model_path saved_model/plate_rec_color_backup.pth --image_path /Users/Downloads/plate
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default=r'color_model/0.989720_epoth_50_model.pth',help='model.pt path(s)')
    parser.add_argument('--model_path', type=str, default=r'saved_model/plate_rec_color.pth',help='model.pt path(s)')
    # parser.add_argument('--image_path', type=str, default=r'/mnt/EPan/carPlate/@realTest2_noTraining/realrealTest/',help='source')
    # parser.add_argument('--image_path', type=str, default=r'images', help='source')
    # parser.add_argument('--image_path', type=str, default=r'/Volumes/Samsung USB/202311-copy', help='source')
    # parser.add_argument('--image_path', type=str, default=r'/Volumes/Samsung USB/small/202312small5', help='source')
    parser.add_argument('--image_path', type=str, default=r'/Volumes/Samsung USB/small/error_plates', help='source')
    # parser.add_argument('--image_path', type=str, default=r'/Volumes/Samsung USB/small/202311small', help='source')
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w', type=int, default=168, help='width')
    parser.add_argument('--LPRNet', action='store_true', help='use LPRNet')  # True代表使用LPRNet ,False代表用plateNet
    parser.add_argument('--acc', type=bool, default='True', help=' get accuracy')  # 标记好的图片，计算准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device =torch.device("cpu")
    opt = parser.parse_args()
    img_size = (opt.img_h, opt.img_w)
    model = init_model(device, opt.model_path)
    if os.path.isfile(opt.image_path):
        # 单张图片
        right = 0
        begin = time.time()
        img = cv_imread(opt.image_path)
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        plate, plate_color = get_plate_result(img, device, model, img_size)
        print(plate, plate_color)
    elif opt.acc:
        # 计算准确率
        file_list = []
        right = 0  # 记录正确识别的车牌数量
        allFilePath(opt.image_path, file_list)  # 收集所有待检测的车牌图像路径
        # 创建一个新的目录来存放识别错误的车牌图像
        error_dir = "error_plates"
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        for pic_ in file_list:
            try:
                pic_name = os.path.basename(pic_)  # 获取图片的文件名
                img = cv_imread(pic_)
                # 确保图像为RGB格式
                if img.shape[-1] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # 使用模型进行车牌识别
                plate, plate_color = get_plate_result(img, device, model, img_size)
                # 从文件名中提取原始车牌号
                plate_ori = pic_.split(os.sep)[-1].split('_')[0]
                # print(pic_,plate_color)
                # print(plate,"---",plate_ori)
                if (plate == plate_ori):  # 比较识别结果与原始车牌号
                    right += 1
                else:
                    # 打印识别错误的信息
                    print(plate_ori, "rec error ---> ", plate, pic_, plate_color)
                    # 将识别错误的车牌图像移动到错误目录
                    shutil.move(pic_, os.path.join(error_dir, pic_name))
                    # print(plate,pic_name)
            except:
                print("error")
        print("sum:%d ,right:%d , accuracy: %f" % (len(file_list), right, right / len(file_list)))
    else:
        # 目录
        file_list = []
        allFilePath(opt.image_path, file_list)
        for pic_ in file_list:
            try:
                pic_name = os.path.basename(pic_)
                img = cv_imread(pic_)
                if img.shape[-1] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                plate, plate_color = get_plate_result(img, device, model)
                print(plate, plate_color, pic_)
            except:
                print("error")
