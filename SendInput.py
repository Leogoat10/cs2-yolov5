# 导入 ctypes 模块，用于与底层的 C 函数进行交互
import ctypes
from ctypes import Union, sizeof, windll  # 从 ctypes 中导入 Union, sizeof, windll
from _ctypes import POINTER, Structure    # 从 _ctypes 中导入 POINTER 和 Structure，用于定义结构体和指针类型

# 定义基本数据类型
LONG = ctypes.c_long       # 定义 LONG 为 ctypes 的 c_long 类型
DWORD = ctypes.c_ulong     # 定义 DWORD 为 ctypes 的 c_ulong 类型
ULONG_PTR = POINTER(DWORD) # 定义 ULONG_PTR 为指向 DWORD 的指针类型

# 定义鼠标输入的结构体
class MOUSEINPUT(Structure):
    _fields_ = (
        ('dx', LONG),           # 水平坐标偏移
        ('dy', LONG),           # 垂直坐标偏移
        ('mouseData', DWORD),   # 鼠标滚轮数据或其他附加数据
        ('dwFlags', DWORD),     # 鼠标事件标志
        ('time', DWORD),        # 时间戳（一般为 0，系统自动生成）
        ('dwExtraInfo', ULONG_PTR)  # 额外信息（用户自定义值或 None）
    )

# 定义联合体，用于在输入结构体中表示鼠标输入
class _INPUTunion(Union):
    _fields_ = (
        ('mi', MOUSEINPUT),  # 鼠标输入
        ('mi', MOUSEINPUT)   # 此处重复用于占位，可以扩展其他输入类型
    )

# 定义输入结构体，用于封装不同类型的输入
class INPUT(Structure):
    _fields_ = (
        ('type', DWORD),          # 输入类型（0 表示鼠标输入）
        ('union', _INPUTunion)    # 联合体，包含鼠标输入等
    )

# 发送输入事件的函数
def SendInput(*inputs):
    """
    调用 WinAPI 函数 SendInput 发送用户输入（如鼠标事件）。
    参数:
        *inputs: 输入事件对象列表。
    返回值:
        发送成功的事件数量。
    """
    nInputs = len(inputs)                 # 获取输入事件的数量
    LPINPUT = INPUT * nInputs             # 创建输入事件数组类型
    pInputs = LPINPUT(*inputs)            # 实例化输入事件数组
    cbSize = ctypes.c_int(sizeof(INPUT))  # 输入结构体的大小
    return windll.user32.SendInput(nInputs, pInputs, cbSize)  # 调用 WinAPI SendInput 函数

# 构造 INPUT 结构体的辅助函数
def Input(structure):
    """
    创建包含特定输入结构体的 INPUT 对象。
    参数:
        structure: 特定输入结构体（如 MOUSEINPUT）。
    返回值:
        包含输入结构体的 INPUT 对象。
    """
    return INPUT(0, _INPUTunion(mi=structure))  # 0 表示鼠标输入类型

# 构造 MOUSEINPUT 结构体的辅助函数
def MouseInput(flags, x, y, data):
    """
    创建 MOUSEINPUT 对象。
    参数:
        flags: 鼠标事件标志（如移动、按下、释放等）。
        x: 鼠标的水平坐标。
        y: 鼠标的垂直坐标。
        data: 附加数据（如滚轮数据）。
    返回值:
        MOUSEINPUT 对象。
    """
    return MOUSEINPUT(x, y, data, flags, 0, None)  # 时间戳和附加信息默认为 0 和 None

# 构造鼠标事件的辅助函数
def Mouse(flags, x=0, y=0, data=0):
    """
    创建鼠标事件的 INPUT 对象。
    参数:
        flags: 鼠标事件标志。
        x: 鼠标的水平坐标（默认值为 0）。
        y: 鼠标的垂直坐标（默认值为 0）。
        data: 附加数据（默认值为 0）。
    返回值:
        包含鼠标事件的 INPUT 对象。
    """
    return Input(MouseInput(flags, x, y, data))

# 模拟鼠标移动事件
def mouse_xy(x, y):
    """
    模拟鼠标移动到指定位置。
    参数:
        x: 鼠标的目标水平坐标。
        y: 鼠标的目标垂直坐标。
    返回值:
        发送成功的事件数量。
    """
    return SendInput(Mouse(0x0001, x, y))  # 0x0001 表示鼠标移动事件

# 模拟鼠标按下事件
def mouse_down(key=1):
    """
    模拟鼠标按下事件。
    参数:
        key: 鼠标按键类型（1 表示左键，2 表示右键）。
    返回值:
        发送成功的事件数量。
    """
    if key == 1:
        return SendInput(Mouse(0x0002))  # 0x0002 表示鼠标左键按下
    elif key == 2:
        return SendInput(Mouse(0x0008))  # 0x0008 表示鼠标右键按下

# 模拟鼠标释放事件
def mouse_up(key=1):
    """
    模拟鼠标释放事件。
    参数:
        key: 鼠标按键类型（1 表示左键，2 表示右键）。
    返回值:
        发送成功的事件数量。
    """
    if key == 1:
        return SendInput(Mouse(0x0004))  # 0x0004 表示鼠标左键释放
    elif key == 2:
        return SendInput(Mouse(0x0010))  # 0x0010 表示鼠标右键释放
