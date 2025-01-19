# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics.utils.ops import scale_coords
from ultralytics.utils.plotting import Annotator
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import set_logging, xyxy2xywh, print_args, non_max_suppression,scale_boxes
from SendInput import *
import pynput
from pynput.mouse import Listener
from screen import screenshot

# 定义文件路径和根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将 ROOT 添加到系统路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转为相对路径

# 定义一个全局变量，表示自瞄是否启用
is_useful = False
# 鼠标点击事件的回调函数
def mouse_click(x, y, button, pressed):
    """
    按下侧键 X2 时启用自瞄，释放侧键 X2 时关闭自瞄。
    """
    global is_useful
    if pressed and button == pynput.mouse.Button.x2:  # 检测是否按下鼠标侧键 X2
        is_useful = True  # 启用自瞄
    elif not pressed and button == pynput.mouse.Button.x2:  # 检测是否释放鼠标侧键 X2
        is_useful = False  # 禁用自瞄
# 鼠标监听器
def mouse_listenr():
    """
    监听鼠标点击事件，用于切换自瞄状态。
    """
    with Listener(on_click=mouse_click) as listener:
        listener.join()

# YOLOv5 推理主程序
@torch.no_grad()
def run():
    """
    启动自瞄逻辑，包括目标检测和鼠标移动模拟。
    """
    # 初始化设备和模型
    device = torch.device('cuda:0')  # 使用 GPU 进行加速
    model = DetectMultiBackend(
        weights='runs/train/exp/weights/best.pt',  # 模型权重路径
        device=device,  # 推理设备
        dnn=False,
        data=False,
        fp16=True  # 启用 FP16 加速
    )

    while True:
        # 截取当前屏幕
        img = screenshot()
        im0 = img.copy()  # 保留原始截图

        # 对截图进行预处理
        img = letterbox(img, (640, 640), stride=32, auto=True)[0]  # 调整为符合模型输入的大小
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(model.device)  # 转为 Tensor 并发送到设备
        img = img.half() if model.fp16 else img.float()  # 转为 float16 或 float32
        img /= 255.0  # 归一化

        if len(img.shape) == 3:
            img = img[None]  # 扩展为 batch 维度

        # 模型推理
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000
        )

        # 处理检测结果
        for i, det in enumerate(pred):  # 遍历每张图片的检测结果
            annotator = Annotator(im0, line_width=1)  # 初始化注释器
            if len(det):  # 如果有检测到目标
                distance_list = []
                target_list = []

                # 将检测框从模型尺寸映射到原始图片尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (
                        xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                    ).view(-1).tolist()  # 转为中心点坐标和宽高
                    #X,Y表示相对截图框原点（左上角）的距离 X1,Y1表示相对于截图框中心（游戏准星）的XY方向距离
                    # X = xywh[0]
                    # Y = xywh[1]
                    X1 = xywh[0] - 200
                    Y1 = xywh[1] - 200
                    #print(X,Y,X1,Y1) 打印坐标用于观察
                    distance = math.sqrt(X1**2 + Y1**2)  # 计算距离
                    xywh.append(distance)
                    # 绘制检测框
                    annotator.box_label(
                        xyxy, label=f'[{int(cls)}]', color=(255, 0, 0), txt_color=(255, 255, 255)
                    )
                    distance_list.append(distance)  # 距离列表
                    target_list.append(xywh)  # 目标信息列表
                # 获取最近的目标信息
                target_info = target_list[distance_list.index(min(distance_list))]
                if is_useful:  # 如果自瞄启用
                    mouse_xy(int(target_info[0] - 200), int(target_info[1] - 200))  # 移动鼠标
                    time.sleep(0.0025)
            # 显示结果
            im0 = annotator.result()
            cv2.imshow('test', im0)
            cv2.setWindowProperty('test', cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)

if __name__ == "__main__":
    # 使用多线程启动鼠标监听器
    threading.Thread(target=mouse_listenr).start()
    run()