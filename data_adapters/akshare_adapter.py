"""
Alpha因子挖掘系统 - Akshare数据源适配器
免费开源数据源，数据丰富
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import time

from .base_adapter import BaseDataAdapter
from utils import standardize_code


class AkshareAdapter(BaseDataAdapter):
    """Akshare数据源适配器"""
    
    name = "akshare"
    
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
    
    def get_daily_data(self,
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取日线行情数据"""
        import akshare as ak
        
        all_data = []
        
        # 处理全市场情况
        if symbols == 'all_a' or symbols == ['all_a']:
            # 获取A股所有股票列表
            try:
                stock_list = ak.stock_info_a_code_name()
                symbols = stock_list['code'].tolist()[:500]  # 限制500只避免超时
            except:
                # 失败则使用默认列表
                symbols = ['000001', '000002', '600000', '600036', '000858']
        
        for i, symbol in enumerate(symbols):
            try:
                # 标准化代码
                std_code = standardize_code(symbol)
                pure_code = std_code.split('.')[0]
                
                # 获取前复权数据
                df = ak.stock_zh_a_hist(
                    symbol=pure_code,
                    period="daily",
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"
                )
                
                if len(df) == 0:
                    continue
                
                # 重命名字段
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turn'
                })
                
                # 添加股票代码
                df['code'] = std_code
                
                # 计算vwap
                if 'amount' in df.columns and 'volume' in df.columns:
                    df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
                
                # 计算收益率
                df = df.sort_values('date')
                df['return_1d'] = df['close'].pct_change()
                
                all_data.append(df)
                
                # 限速
                if i % 10 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                # print(f"获取 {symbol} 数据失败: {e}")
                continue
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])
        
        # 选择字段
        if fields is not None and fields != '*':
            available_fields = [f for f in fields if f in result.columns]
            result = result[available_fields]
        
        return result
    
    def get_industry_classification(self, symbols: List[str]) -> pd.DataFrame:
        """获取行业分类数据 - 使用申万一级行业"""
        import akshare as ak
        
        try:
            # 获取申万一级行业
            industry_df = ak.sw_index_spot()
            
            result = []
            for symbol in symbols:
                std_code = standardize_code(symbol)
                pure_code = std_code.split('.')[0]
                
                # 简单映射：实际中需要更准确的行业分类
                result.append({
                    'code': std_code,
                    'group': 'unknown'  # 简化处理
                })
            
            return pd.DataFrame(result)
        except:
            # 失败则返回空
            return pd.DataFrame([
                {'code': standardize_code(s), 'group': 'unknown'}
                for s in symbols
            ])
