"""
Alpha因子挖掘系统 - 统一数据抽象层
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

from data_adapters import (
    BaseDataAdapter,
    AkshareAdapter,
    TushareAdapter,
    BaostockAdapter,
    YFinanceAdapter
)
from utils import ensure_dir, save_parquet_cache, load_parquet_cache, get_cache_key


class DataHub:
    """统一数据入口类"""
    
    ADAPTER_MAP = {
        'akshare': AkshareAdapter,
        'tushare': TushareAdapter,
        'baostock': BaostockAdapter,
        'yfinance': YFinanceAdapter,
    }
    
    def __init__(self,
                 preferred_source: str = 'akshare',
                 fallback_sources: List[str] = None,
                 cache_dir: str = './data_cache',
                 start_date: str = '2018-01-01',
                 end_date: str = '2025-01-01',
                 tushare_token: str = None):
        """
        初始化数据中心
        
        Args:
            preferred_source: 首选数据源
            fallback_sources: 降级数据源列表
            cache_dir: 本地缓存目录
            start_date: 数据开始日期
            end_date: 数据结束日期
            tushare_token: Tushare Token
        """
        self.preferred_source = preferred_source
        self.fallback_sources = fallback_sources or ['tushare', 'baostock']
        self.cache_dir = ensure_dir(cache_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.tushare_token = tushare_token
        
        # 初始化适配器
        self._adapters: Dict[str, BaseDataAdapter] = {}
        self._init_adapters()
        
        # 缓存数据
        self._cached_data = None
    
    def _init_adapters(self):
        """初始化所有数据源适配器"""
        for source in [self.preferred_source] + self.fallback_sources:
            if source in self.ADAPTER_MAP:
                adapter_cls = self.ADAPTER_MAP[source]
                if source == 'tushare':
                    self._adapters[source] = adapter_cls(token=self.tushare_token)
                else:
                    self._adapters[source] = adapter_cls()
    
    def _get_cache_path(self, symbols, start_date, end_date, source: str) -> str:
        """获取缓存文件路径"""
        cache_key = get_cache_key(
            symbols=str(symbols),
            start=start_date,
            end=end_date,
            source=source
        )
        return os.path.join(self.cache_dir, f"data_{cache_key}.parquet")
    
    def get_daily_data(self,
                       symbols = 'all_a',
                       fields = '*',
                       use_cache: bool = True) -> pd.DataFrame:
        """
        获取全市场日线数据
        
        Args:
            symbols: 股票代码列表或'all_a'表示全部A股
            fields: 字段列表或'*'表示全部
            use_cache: 是否使用缓存
        
        Returns:
            标准格式的DataFrame
        """
        if use_cache and self._cached_data is not None:
            return self._cached_data
        
        # 尝试从缓存加载
        cache_path = self._get_cache_path(symbols, self.start_date, self.end_date, 'combined')
        if use_cache and os.path.exists(cache_path):
            print(f"[DataHub] 从缓存加载数据: {os.path.basename(cache_path)}")
            df = load_parquet_cache(cache_path)
            self._cached_data = df
            return df
        
        # 依次尝试数据源
        all_sources = [self.preferred_source] + self.fallback_sources
        df = None
        for source in all_sources:
            if source not in self._adapters:
                continue
            
            adapter = self._adapters[source]
            if not adapter.is_available():
                print(f"[DataHub] 尝试从 {source} 拉取数据...")
                try:
                    df = adapter.get_daily_data(
                        symbols=symbols,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        fields=fields
                    )
                    if len(df) > 0:
                        print(f"[DataHub] 从 {source} 成功获取 {len(df)} 条数据")
                        break
                except Exception as e:
                    print(f"[DataHub] {source} 拉取失败: {e}")
                    continue
        
        if df is None or len(df) == 0:
            print("[DataHub] 所有数据源均失败，使用模拟数据")
            df = self._generate_mock_data(symbols)
        
        # 计算衍生特征
        df = self.prepare_features(df)
        
        # 设置多级索引
        df = df.set_index(['date', 'code']).sort_index()
        
        # 保存缓存
        if use_cache:
            save_parquet_cache(df.reset_index(), cache_path)
            self._cached_data = df
        
        return df
    
    def _generate_mock_data(self, symbols) -> pd.DataFrame:
        """生成模拟数据"""
        if symbols == 'all_a' or symbols == ['all_a']:
            symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', 
                      '000858.SZ', '600519.SH', '000333.SZ', '002415.SZ',
                      '000568.SZ', '000651.SZ']
        
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        all_data = []
        np.random.seed(42)
        
        for symbol in symbols:
            # 生成模拟价格
            base_price = 10 + np.random.random() * 100
            
            # 随机游走
            returns = np.random.normal(0.001, 0.02, len(dates))
            price_returns[0] = 0
            prices = base_price * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'date': dates,
                'code': symbol,
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates)),
                'amount': prices * np.random.randint(1000000, 10000000, len(dates)),
                'turn': np.random.uniform(0.01, 0.1, len(dates)),
            })
            
            # 修正高低价
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算衍生特征列
        
        Args:
            df: 原始行情数据
        
        Returns:
            添加衍生列后的DataFrame
        """
        df = df.copy()
        
        # 收益率
        df['return_1d'] = df.groupby('code')['close'].pct_change()
        df['return_5d'] = df.groupby('code')['close'].pct_change(5)
        df['return_20d'] = df.groupby('code')['close'].pct_change(20)
        
        # 波动率
        df['volatility_20d'] = df.groupby('code')['return_1d'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        
        # 高低价比
        df['high_low_ratio'] = df['high'] / df['low']
        
        # vwap相关
        if 'vwap' not in df.columns:
            df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
        df['close_vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
        
        # 成交量相关
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # 换手率均线
        if 'turn' in df.columns:
            df['turn_5d_avg'] = df.groupby('code')['turn'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
        
        # 市值代理
        df['market_cap'] = df['close'] * 1e9  # 简化
        df['log_market_cap'] = np.log(df['market_cap'])
        
        # 行业分类
        if 'group' not in df.columns:
            df['group'] = 'unknown'
        
        return df
