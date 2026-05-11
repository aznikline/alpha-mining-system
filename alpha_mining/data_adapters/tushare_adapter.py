"""
Alpha因子挖掘系统 - Tushare数据源适配器
专业金融数据源，数据质量高，需token
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .base_adapter import BaseDataAdapter
from ..utils import standardize_code


class TushareAdapter(BaseDataAdapter):
    """Tushare数据源适配器"""
    
    name = "tushare"
    priority = 1  # A股备选
    market_support = ['a_share']
    
    def __init__(self, token: str = None):
        self.token = token
        self._api = None
        self._available = None
    
    def _get_api(self):
        if self._api is None and self.token:
            import tushare as ts
            ts.set_token(self.token)
            self._api = ts.pro_api()
        return self._api
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import tushare
                self._available = self.token is not None
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
        api = self._get_api()
        if api is None:
            return pd.DataFrame()
        
        all_data = []
        
        if symbols == 'all' or symbols == ['all']:
            symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        
        for symbol in symbols:
            try:
                std_code = standardize_code(symbol)
                ts_code = std_code
                
                df = api.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                
                if len(df) == 0:
                    continue
                
                df = df.rename(columns={'trade_date': 'date', 'pct_chg': 'return_1d'})
                df['code'] = std_code
                df['date'] = pd.to_datetime(df['date'])
                df['return_1d'] = df['return_1d'] / 100
                
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
