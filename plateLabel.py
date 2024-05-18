import cv2
# import imageio
import numpy as np
import os
import shutil
import argparse
from alphabets import plate_chr  # 导入车牌可能出现的所有字符

# 遍历rootfile文件下所有图片
def allFileList(rootfile, allFile):
    """
    遍历指定根目录下的所有文件和子目录，将所有文件的路径添加到指定的列表中。

    :param rootfile: 根目录的文件路径。
    :param allFile: 用于收集所有文件路径的列表。
    """
    folder = os.listdir(rootfile)  # 列出根目录下的所有文件和子目录
    for temp in folder:
        fileName = os.path.join(rootfile, temp)  # 拼接当前文件或目录的完整路径
        if os.path.isfile(fileName):  # 如果是文件，则添加到allFile列表中
            allFile.append(fileName)
        else:  # 如果是目录，则递归调用allFileList继续遍历该目录下的文件和子目录
            allFileList(fileName, allFile)



# 判断车牌名是不是在palteStr中  当车牌名不在plateStr中的 return False
def is_str_right(plate_name):
    for str_ in plate_name:
        if str_ not in palteStr:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/mnt/Gu/trainData/plate/final", help='source')
    parser.add_argument('--label_file', type=str, default='datasets/train.txt', help='model.pt path(s)')

    opt = parser.parse_args()
    rootPath = opt.image_path
    labelFile = opt.label_file
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危险品0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    # plate_chr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
    palteStr = plate_chr
    print(len(palteStr))

    # 生成一个字典plateDict plateDict={'#':0, '京':1, '沪':2 ......}
    plateDict = {}
    for i in range(len(list(palteStr))):
        plateDict[palteStr[i]] = i
    fp = open(labelFile, "w", encoding="utf-8")
    file = []
    allFileList(rootPath, file)  # 遍历rootPath下所有图片  保存在file中
    picNum = 0

    # 遍历每一张图片
    for jpgFile in file:
        print(jpgFile)
        jpgName = os.path.basename(jpgFile)  # 获得图片名称  如: 云A008BC_0.jpg
        name = jpgName.split("_")[0]  # 获得车牌文字pstr  如: 云A008BC
        if " " in name:
            continue
        labelStr = " "
        if not is_str_right(name):  # 如果车牌文字pstr存在不在plateDict中的字符pchar 则直接continue
            continue
        strList = list(name)  # 将车牌文字转化为列表 如: ['云','A','0','0','8','B','C']
        # p_number表示车牌字符对应的数字, 即plateDict中的0
        for i in range(len(strList)):
            labelStr += str(plateDict[strList[i]]) + " "  # 将车牌文字转化为对应的数字p_number 如: "25 52 42 42 50 53 54"
        # while i<7:
        #     labelStr+=str(0)+" "
        #     i+=1
        picNum += 1
        # print(jpgFile+labelStr)
        # 将图片路径和对应的标签写入labelFile中  如 datasets/val\云A008BC_0.jpg 25 52 42 42 50 53 54
        # 如：/content/drive/MyDrive/u_train/recognition_dir/val_verify/渝G12775_0.jpg 4 58 43 44 49 49 47
        fp.write(jpgFile + labelStr + "\n")
    fp.close()
