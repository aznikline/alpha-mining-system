"""
Alpha因子挖掘系统 - Tushare数据源适配器
专业金融数据源，数据质量高
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .base_adapter import BaseDataAdapter
from utils import standardize_code


class TushareAdapter(BaseDataAdapter):
    """Tushare数据源适配器"""
    
    name = "tushare"
    
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
    
    def get_daily_data(self,
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取日线行情数据"""
        api = self._get_api()
        if api is None:
            return pd.DataFrame()
        
        all_data = []
        
        if symbols == 'all_a' or symbols == ['all_a']:
            symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        
        for symbol in symbols:
            try:
                std_code = standardize_code(symbol)
                ts_code = std_code.replace('.SZ', '.SZ').replace('.SH', '.SH')
                
                df = api.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                
                if len(df) == 0:
                    continue
                
                # 重命名字段
                df = df.rename(columns={
                    'trade_date': 'date',
                    'pct_chg': 'return_1d'
                })
                
                df['code'] = std_code
                df['date'] = pd.to_datetime(df['date'])
                df['return_1d'] = df['return_1d'] / 100
                
                # 计算vwap
                if 'amount' in df.columns and 'vol' in df.columns:
                    df['vwap'] = df['amount'] / (df['vol'] * 100).replace(0, np.nan)
                
                all_data.append(df)
                
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        return result
    
    def get_industry_classification(self, symbols: List[str]) -> pd.DataFrame:
        """获取行业分类"""
        api = self._get_api()
        if api is None:
            return pd.DataFrame([
                {'code': standardize_code(s), 'group': 'unknown'}
                for s in symbols
            ])
        
        try:
            industry_list = []
            for symbol in symbols:
                std_code = standardize_code(symbol)
                industry_list.append({
                    'code': std_code,
                    'group': 'unknown'
                })
            return pd.DataFrame(industry_list)
        except:
            return pd.DataFrame([
                {'code': standardize_code(s), 'group': 'unknown'}
                for s in symbols
            ])
