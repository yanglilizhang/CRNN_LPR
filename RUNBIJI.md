

$ pip install pyyaml -i https://mirrors.aliyun.com/pypi/simple/
$ pip install easydict -i https://mirrors.aliyun.com/pypi/simple/
$ pip install tensorboard -i https://mirrors.aliyun.com/pypi/simple/
安装pytorch cpu版 https://pytorch.org/get-started/previous-versions/
$ conda install pytorch::pytorch torchvision torchaudio -c pytorch
$ pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/

1.解析数据集打上标签,生成train.txt和val.txt的程序
python plateLabel.py --image_path your/train/img/path/ --label_file datasets/train.txt
python plateLabel.py --image_path your/val/img/path/ --label_file datasets/val.txt

https://blog.csdn.net/shilichangtin/article/details/135005410?spm=1001.2014.3001.5502 代码解析
https://blog.csdn.net/shilichangtin/article/details/134984373 yolov7

$ python demo.py --model_path saved_model/plate_rec_color.pth --image_path images/test.jpg
输出车牌号
出现错误：问题是模型训练是是GPU,当前运行是CPU.
修改demo中：model.load_state_dict(model_state,strict=False) 新增strict=False
上面是load_state_dict方法参数的官方说明 strict  参数默认是true，他的含义是 是否严格要求state_dict中的键与该模块的键返回的键匹配
就是说，如果strict 置为false那么就可以忽略掉报错，请注意是忽略哦！！！


train.py
config = yaml.load(f, Loader=yaml.FullLoader)
# config = yaml.load(f)

Traceback (most recent call last):
  File "demo.py", line 84, in <module>
    model = init_model(device,opt.model_path)
  File "demo.py", line 67, in init_model
    model.load_state_dict(model_state)
  File "/Users/zhangwei/anaconda3/envs/crnnpy37/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1672, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for myNet_ocr:
        Unexpected key(s) in state_dict: "conv1.weight", "conv1.bias", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked", "color_classifier.weight", "color_classifier.bias", 

$ python demo_plate_color.py --model_path saved_model/plate_rec_color.pth --image_path images/test.jpg
输出车牌及颜色


$ pip install tensorboardX -i https://mirrors.aliyun.com/pypi/simple/


不能保证长度是固定
img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
img = cv2.resize(img, (self.input_w,self.input_h))

def getitem(self, idx):
统一宽度，不够400，填充黑色；
bk = np.ones(shape=[self.input_h,400,3], dtype=np.uint8)
#bk = cv2.Mat(cv2.Size(400,self.input_h),cv2.CV_8UC3)
dstw = (int)(img_w*self.input_h/img_h)
img = cv2.resize(img, (dstw,self.input_h))
bk[0:self.input_h,0:dstw] = img
#roim = bk[0:dstw,0:self.input_h]			
#roim = img
cv2.imwrite("out1.jpg",img)
img = bk
cv2.imwrite("out.jpg",img)
