"""
Alpha因子挖掘系统 - 工具函数
"""
import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def standardize_code(code: str) -> str:
    """
    股票代码标准化为 代码.交易所 格式
    支持：000001, 000001.SZ, sh000001 等多种格式
    """
    code = str(code).strip().upper()
    
    # 去除前缀
    if code.startswith('SH') or code.startswith('SZ'):
        code = code[2:]
    if code.startswith('S') and len(code) > 6:
        code = code[1:]
    
    # 提取纯数字部分
    import re
    match = re.search(r'(\d{6})', code)
    if match:
        pure_code = match.group(1)
    else:
        pure_code = code[:6] if len(code) >= 6 else code.zfill(6)
    
    # 判断交易所
    if pure_code[0] in ['6', '5', '9', '11']:
        return f"{pure_code}.SH"
    elif pure_code[0] in ['0', '3', '1', '2', '12']:
        return f"{pure_code}.SZ"
    else:
        return f"{pure_code}.SH"  # 默认上海


def get_cache_key(*args, **kwargs) -> str:
    """生成缓存键名"""
    key_str = '_'.join(str(arg) for arg in args)
    key_str += '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def save_parquet_cache(df: pd.DataFrame, path: str):
    """保存Parquet缓存"""
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, compression='snappy')


def load_parquet_cache(path: str) -> pd.DataFrame or None:
    """加载Parquet缓存"""
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """生成日期范围"""
    return pd.date_range(start=start_date, end=end_date, freq='B')


def fill_missing_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """填充缺失的交易日"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    all_dates = date_range(min_date, max_date)
    date_df = pd.DataFrame({date_col: all_dates})
    
    return date_df.merge(df, on=date_col, how='left')


def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """缩尾处理，去除极端值"""
    q_low = x.quantile(lower)
    q_high = x.quantile(upper)
    return x.clip(q_low, q_high)


def progress_bar(current: int, total: int, prefix: str = '', length: int = 30):
    """打印进度条"""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} |{bar}| {current}/{total} ({percent*100:.1f}%)', end='', flush=True)
    if current == total:
        print()
