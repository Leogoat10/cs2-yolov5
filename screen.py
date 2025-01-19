import cv2
from mss import mss
import numpy as np

# 获取屏幕分辨率
ScreenX = 1920
ScreenY = 1080

# 定义窗口大小
window_width = 400
window_height = 400

# 计算中心位置
window_size = {
    "left": int((ScreenX - window_width) / 2),  # 屏幕中心的左上角 X 坐标
    "top": int((ScreenY - window_height) / 2), # 屏幕中心的左上角 Y 坐标
    "width": window_width,  # 窗口宽度
    "height": window_height # 窗口高度
}
# 初始化 mss
Screenshot_value = mss()

def screenshot():
    # 截取屏幕中心区域
    img = Screenshot_value.grab(window_size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return  img  # 转换为 numpy 数组

# 截图并显示
# while True:
#     cv2.imshow('Center Screenshot', screenshot())
#     cv2.setWindowProperty('Center Screenshot', cv2.WND_PROP_TOPMOST, 1)
#     cv2.waitKey(1)
