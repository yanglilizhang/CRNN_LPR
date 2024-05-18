import argparse
# from plateNet import myNet_ocr
from colorNet import myNet_ocr_color
from alphabets import plate_chr
import torch
import onnx


# 导出颜色分支训练出来的模型为onnx
# python export.py --weights saved_model/plate_rec_color.pth --save_path saved_model/plate_rec_color.onnx  --simplify
if __name__=="__main__":
    """
        将训练好的PyTorch识别模型(字符和颜色)转换为ONNX格式。

        参数:
        - opt: 包含模型权重路径、批量大小、保存路径等配置信息的对象。

        返回值:
        - 无返回值，但会生成ONNX模型文件和（根据配置）简化后的ONNX模型文件。
        """
    parser=argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='saved_model/best.pth', help='weights path')  # from yolov5/models/
    parser.add_argument('--weights', type=str, default='saved_model/plate_rec_color.pth', help='weights path')  # from yolov5/models/
    parser.add_argument('--save_path', type=str, default='plate_rec_color.onnx', help='onnx save path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[48, 168], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--simplify', action='store_true', default=False, help='simplified onnx')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print(opt)

    # 从指定路径加载模型权重
    checkpoint = torch.load(opt.weights, map_location=device)
    cfg = checkpoint['cfg']  # 从检查点中提取配置信息

    # 实例化模型，配置出口并设定颜色数量
    model = myNet_ocr_color(num_classes=len(plate_chr),cfg=cfg,export=True,color_num=5)
    model.load_state_dict(checkpoint['state_dict']) # 加载权重
    model.eval() # 设定模型为评估模式
    
    input = torch.randn(opt.batch_size,3,48,168)
    onnx_file_name = opt.save_path
    
    torch.onnx.export(model,input,onnx_file_name,
                      input_names=["images"],output_names=["output_1","output_2"],
                      verbose=False,
                      opset_version=11,
                      dynamic_axes={'images': {0: 'batch'},
                                    'output': {0: 'batch'}
                                   } if opt.dynamic else None)
    print(f"convert completed,save to {opt.save_path}")                  
    if opt.simplify:
        from onnxsim import simplify
        print(f"begin simplify ....")
        input_shapes = {"images": list(input.shape)}
        onnx_model = onnx.load(onnx_file_name)
        model_simp, check = simplify(onnx_model,test_input_shapes=input_shapes)
        onnx.save(model_simp, onnx_file_name)
        print(f"simplify completed,save to {opt.save_path}")
    
    