"""
Alpha因子挖掘系统 - Akshare数据源适配器
免费开源数据源，覆盖A股
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import time

from .base_adapter import BaseDataAdapter
from ..utils import standardize_code


class AkshareAdapter(BaseDataAdapter):
    """Akshare数据源适配器"""
    
    name = "akshare"
    priority = 0  # A股首选
    market_support = ['a_share']  # 仅支持A股
    
    def __init__(self):
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import akshare
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def fetch(self,
             market: str,
             symbols: List[str],
             start_date: str,
             end_date: str,
             fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取日线行情数据"""
        import akshare as ak
        
        all_data = []
        
        # 处理全市场
        if symbols == 'all' or symbols == ['all']:
            # 简化：只拉取沪深300成分股作为演示
            symbols = ['000001', '000002', '600000', '600036', '000858',
                      '600519', '000333', '002415', '000568', '000651']
        
        for i, symbol in enumerate(symbols):
            try:
                std_code = standardize_code(symbol)
                pure_code = std_code.split('.')[0]
                
                df = ak.stock_zh_a_hist(
                    symbol=pure_code,
                    period="daily",
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"  # 前复权
                )
                
                if len(df) == 0:
                    continue
                
                # 字段映射
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turn',
                })
                
                df['code'] = std_code
                all_data.append(df)
                
                if i % 10 == 9:
                    time.sleep(0.3)
                    
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        return result
    
    def get_industry_classification(self, market: str, symbols: List[str]) -> pd.DataFrame:
        """获取行业分类"""
        try:
            # 简化实现，返回未知
            return pd.DataFrame([
                {'code': standardize_code(s), 'group': 'unknown'}
                for s in symbols
            ])
        except:
            return pd.DataFrame([
                {'code': standardize_code(s), 'group': 'unknown'}
                for s in symbols
            ])
