"""
Alpha因子挖掘系统 - Baostock数据源适配器
免费，数据稳定，覆盖A股
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .base_adapter import BaseDataAdapter
from utils import standardize_code


class BaostockAdapter(BaseDataAdapter):
    """Baostock数据源适配器"""
    
    name = "baostock"
    priority = 2  # A股降级备选
    market_support = ['a_share']
    
    def __init__(self):
        self._available = None
        self._lg = None
    
    def _login(self):
        if self._lg is None:
            import baostock as bs
            self._lg = bs.login()
        return self._lg
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import baostock
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
        import baostock as bs
        self._login()
        
        all_data = []
        
        if symbols == 'all' or symbols == ['all']:
            symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        
        for symbol in symbols:
            try:
                std_code = standardize_code(symbol)
                bs_code = 'sz.' + std_code.split('.')[0] if 'SZ' in std_code else 'sh.' + std_code.split('.')[0]
                
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,turn",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2"  # 前复权
                )
                
                df = rs.get_data()
                
                if len(df) == 0:
                    continue
                
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['code'] = std_code
                df['date'] = pd.to_datetime(df['date'])
                all_data.append(df)
                
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        return result
    
    def get_industry_classification(self, market: str, symbols: List[str]) -> pd.DataFrame:
        """获取行业分类"""
        return pd.DataFrame([
            {'code': standardize_code(s), 'group': 'unknown'}
            for s in symbols
        ])
