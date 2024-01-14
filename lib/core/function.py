from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # 注意这里的idx还是索引, 可以从上面的数据集读取上看到return idx
        # labels: ['苏A8C4A8', '川AE00K0', '冀EL2392', '鲁ATN619', '川A5E1Z9', '闽FQZ790', '辽MD7792'......] len=256
        labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)

        # inference
        # preds: torch.Size([21, 256, 78])
        # 21: 车牌预测的字符个数的最大上限 也就是一张车牌最多预测21个字符pchar
        # 256：图片的batchsize
        # 78：车牌字符集：77 + 1 (77是车牌字符串plate_name的长度 1是blank， 相当于#)
        # 这个空白#的存在其实和CTCLoss有关 这里就不过多介绍了
        preds = model(inp).cpu()

        # 计算损失
        # batchsize: 256
        batch_size = inp.size(0)
        # text:  tensor([11, 52, 50,  ..., 47, 42, 47], dtype=torch.int32)  shape: torch.Size([1798])
        # length:  tensor([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, ......]  shape:torch.Size([256])
        # 从上面的输出结果可以看出text中的是将labels中所有的车牌字符串pstr(pstr表示每一个车牌字符串 如'云A008BC')都拼接在一起, 其中的值代表的是每一个plate_chr{#京沪......}对应的下标
        # length中的值则可以很轻松的看出是每一个车牌pstr(pstr表示每一个车牌字符串 如'云A008BC')的长度
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        # preds_size:  tensor([21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, ......]  shape:torch.Size([256])
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        # torch官网上的CTCLoss的使用的参数要求 可以直接看官网 官网更详细
        # preds:  (T, N, C)  T=input length  N=batch size  C=number of classes(including blank)
        # text: (N, S) or (sum(target_lengths)) sum就是将所有的字符串pstr拼接在一起并转化为对应的plateDict下标  其中0是blank
        # preds_size: (N, )
        # length: (N, )
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    sum = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            # preds: shape:(21, 128, 78)
            # 在max之后 _: 是最大值 shape(21, 128)  preds: 是最大值的索引 shape(21, 128)
            # 这个部分主要是选出车牌字符串集plateDict 78中最大概率的那个作为该位置的输出
            _, preds = preds.max(2)

            # preds: torch.Size([2688])
            # 先转化为[128, 21], 主要是为了decode中每一个相邻的21个位置都是同一个车牌上的预测结果
            preds = preds.transpose(1, 0).contiguous().view(-1)
            # preds: tensor([30,  0,  0,  ..., 43, 43,  0])  torch.Size([2688])
            # preds_size: tensor([21, 21, 21, ......])  shape: torch.Size([128])
            # 这里的converter.decode是跟CTCLoss进行配合的, 需要好好理解一下
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            # 验证集指标的出现
            for pred, target in zip(sim_preds, labels):
                sum+=1
                if pred == target:
                    n_correct += 1

            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(config.TEST.NUM_TEST* config.TEST.BATCH_SIZE_PER_GPU)
    # accuracy = n_correct / float(config.TEST.NUM_TEST * config.TEST.BATCH_SIZE_PER_GPU)
    # 这个指标是完全预测准确的车牌/总的预测的车牌
    accuracy = n_correct / sum
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy