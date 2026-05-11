from .base_adapter import BaseDataAdapter
from .akshare_adapter import AkshareAdapter
from .tushare_adapter import TushareAdapter
from .baostock_adapter import BaostockAdapter
from .yfinance_adapter import YFinanceAdapter

__all__ = [
    'BaseDataAdapter',
    'AkshareAdapter',
    'TushareAdapter',
    'BaostockAdapter',
    'YFinanceAdapter',
]

# 按市场分类的适配器
MARKET_ADAPTERS = {
    'a_share': [AkshareAdapter, TushareAdapter, BaostockAdapter],
    'us': [YFinanceAdapter],
}
