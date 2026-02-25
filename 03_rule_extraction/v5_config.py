"""
v5_config.py — 共享常量与 matplotlib 全局配置
"""

import matplotlib
matplotlib.use('Agg')  # headless backend, faster rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Matplotlib 配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 背景EC：太常见、缺乏区分度的EC类
BACKGROUND_EC = {'1.1.1'}

# 载体分子黑名单（高频中间节点，会污染路径）
UBIQUITOUS_COMPOUNDS = {
    # 水 / 质子 / 简单无机物
    "C00001", "C00007", "C00080", "C00011", "C00014", "C00059", "C00283",
    # 磷酸 / 能量载体
    "C00002", "C00008", "C00020", "C00013", "C00009",
    "C00044", "C00035", "C00075", "C00015", "C00063", "C00112",
    # 氧化还原辅酶
    "C00003", "C00004", "C00005", "C00006",
    "C00016", "C01352", "C00061", "C01847",
    # 辅酶A相关
    "C00010", "C00024", "C00083", "C00091",
    # 其他常见载体
    "C00019", "C00021", "C00101", "C00234", "C00440", "C00053", "C00054",
}


def to_ec_level(ec: str, level: int = 3) -> str:
    """截取 EC 编号到指定级别 (e.g. '1.14.14.130' → '1.14.14')"""
    if not ec or '.' not in ec:
        return ''
    parts = ec.split('.')
    if len(parts) < level:
        return ''
    result_parts = parts[:level]
    if result_parts[-1] == '-':
        return ''
    return '.'.join(result_parts)
