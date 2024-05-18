import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
from lib.utils.utils import model_info
from plateNet import myNet_ocr
from alphabets import plateName, plate_chr
from LPRNet import build_lprnet

from tensorboardX import SummaryWriter


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height')  # 模型input的h
    parser.add_argument('--img_w', type=int, default=168, help='width')  # 模型input的w
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        # 将config转化为edict形式的  即从config['DATASET']['ALPHABETS']变成config.DATASET.ALPHABETS']
        config = edict(config)

    # 基础配置部分
    config.DATASET.ALPHABETS = plateName  # 字符集plate_name 比plate_chr少了一个blank字符"#"
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)  # 字符集plate_name长度 77
    config.HEIGHT = args.img_h  # 输入图片的h
    config.WIDTH = args.img_w  # 输入图片的w
    return config


# 训练的时候也需要这样处理，将双层车牌分割，拼接成单层
def main():
    # 加载config
    config = parse_arg()

    # 所有保存文件的输出路径
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn配
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    # cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128] #small model
    cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 256]  # medium model
    # cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256] #big model
    # model = crnn.get_crnn(config,cfg=cfg)
    model = myNet_ocr(num_classes=len(plate_chr), cfg=cfg)
    # model = build_lprnet(num_classes=len(plate_chr))

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        # 如果CUDA可用，将设备设置为指定的GPU
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        # 如果CUDA不可用，使用CPU作为设备
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    # 初始化训练过程的优化器和学习率调度器-根据配置和模型初始化优化器和学习率调度器。
    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    # 根据配置中学习率步骤的类型（列表或单个值），选择不同的学习率调度器
    if isinstance(config.TRAIN.LR_STEP, list):
        # 如果是列表，使用MultiStepLR调度器，它允许在多个特定epoch降低学习率
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        # 如果是单个值，使用StepLR调度器，它会在每个指定epoch降低学习率
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    # 是否进行细调
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        # 加载模型检查点并将其映射到 CPU
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        # from collections import OrderedDict
        # model_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     if 'cnn' in k:
        #         model_dict[k[4:]] = v
        # model.cnn.load_state_dict(model_dict)
        model.load_state_dict(checkpoint)
        # if config.TRAIN.FINETUNE.FREEZE:
        #     for p in model.cnn.parameters():
        #         p.requires_grad = False

    # 是否从指定文件恢复训练
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    # 获取训练数据集
    train_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=True)
    # 初始化数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    # 初始化标签转换器，依据配置文件中指定的字符集
    # 利用{'#': 0, '京': 1, ......}中的一一对应关系 将车牌名字转化为对应的数字
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # 循环遍历训练过程中的每个epoch，从上一个epoch开始直到配置中指定的结束epoch
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # 进行一个训练epoch，更新模型参数
        function.train(config, train_loader, train_dataset, converter, model,
                       criterion, optimizer, device, epoch, writer_dict, output_dict)
        # 调整学习率，基于当前epoch和配置
        lr_scheduler.step()

        # 在验证集上评估模型性能，返回准确率
        acc = function.validate(config, val_loader, val_dataset, converter,
                                model, criterion, device, epoch, writer_dict, output_dict)

        # 判断当前验证集准确率是否为最佳，更新最佳准确率
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "cfg": cfg,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
