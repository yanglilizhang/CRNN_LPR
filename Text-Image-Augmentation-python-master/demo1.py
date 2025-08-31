# -*- coding:utf-8 -*-
# Author: RubanSeven

import cv2
import imageio
import numpy as np
import os
from augment import distort, stretch, perspective
import argparse


def allFileList(rootfile, allFile):
    folder = os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile, temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName, allFile)


def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


# 数据增强
# cd Text-Image-Augmentation-python-master
# python demo1.py --src_path /mnt/Gu/trainData/test_aug --dst_path /mnt/Gu/trainData/result_aug/
# src_path 是数据路径， dst_path是保存的数据路径
# 然后把两份数据放到一起进行训练，效果会好很多！

# Invalid channel number 4 for image: imgs/test.jpg
# 4通道图像通常包含透明度(alpha)通道
#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--src_path', type=str, default='d:/sjzq', help='model.pt path(s)')
    # parser.add_argument('--dst_path', type=str, default='d:/sjzq2', help='source')
    parser.add_argument('--src_path', type=str, default='imgs/', help='model.pt path(s)')
    parser.add_argument('--dst_path', type=str, default='imgs_result/', help='source')
    opt = parser.parse_args()
    rootFile = opt.src_path
    saveFile = opt.dst_path

    # 在开始处理文件之前就创建目标目录
    if not os.path.exists(saveFile):
        os.makedirs(saveFile)

    fileList = []
    allFileList(rootFile, fileList)
    picOunt = 0
    for temp in fileList:
        print(picOunt, temp)
        picOunt += 1
        im = cv2.imdecode(np.fromfile(temp, dtype=np.uint8), -1)
        if im is None:
            print(f"Failed to read image: {temp}")
            continue
        # _,_,c = im.shape
        h, w, c = im.shape
        if c == 1:
            # 灰度图转RGB
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        elif c == 4:
            # RGBA转RGB
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        h, w, c = im.shape
        if c != 3:
            print(f"Invalid channel number {c} for image: {temp}")
            continue
        # im = cv2.resize(im, (200, 64))
        #     cv2.imshow("im_CV", im)
        distort_img_list = list()
        stretch_img_list = list()
        perspective_img_list = list()
        for i in range(1):
            try:
                distort_img = distort(im, 8)
                distort_img_list.append(distort_img)
                # cv2.imshow("distort_img", distort_img)

                stretch_img = stretch(distort_img, 8)
                # cv2.imshow("stretch_img", stretch_img)
                stretch_img_list.append(stretch_img)

                # 在保存文件之前添加目录检查和创建代码
                name = temp.split(os.sep)[-1].split(".")[0]
                name = name + str(picOunt) + ".jpg"
                newPath = os.path.join(saveFile, name)
                # 确保目标目录存在
                if not os.path.exists(saveFile):
                    os.makedirs(saveFile)
                    print(f"Created directory: {saveFile}")

                perspective_img = perspective(stretch_img)
                # cv2.imshow("perspective_img", perspective_img)
                perspective_img_list.append(perspective_img)
                # cv2.waitKey(1)
                cv2.imencode('.jpg', perspective_img)[1].tofile(newPath)
            except Exception as e:
                print(f"Error processing {temp}: {str(e)}")
                continue
    # create_gif(distort_img_list, r'imgs/distort.gif')
    # create_gif(stretch_img_list, r'imgs/stretch.gif')
    # create_gif(perspective_img_list, r'imgs/perspective.gif')
