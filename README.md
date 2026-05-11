# Alpha因子挖掘系统 v1.0

一个模块化、可扩展的自动化Alpha因子挖掘系统，参考WorldQuant和BigQuant的设计思想。

## 核心特性

### 📊 统一数据抽象层
- **多数据源支持**: Akshare（免费推荐）、Tushare（专业）、Baostock（稳定）、YFinance（美股）
- **自动降级机制**: 首选数据源失败时自动尝试备用源
- **本地Parquet缓存**: 重复使用无需重复下载，加速回测
- **增量更新**: 自动检测缓存并拉取最新数据
- **数据标准化**: 统一的字段命名、格式和前复权处理

### 🔧 因子引擎
- **表达式解析**: 支持字符串表达式直接计算，如 `rank(ts_mean(close, 20) / close)`
- **内置算子库**: 截面算子（rank, zscore）、时序算子（ts_mean, ts_std, ts_corr）、逻辑算子
- **批量并行计算**: 支持同时计算多个因子
- **因子值缓存**: Joblib序列化缓存，避免重复计算

### 📈 因子评估体系
- **IC分析**: Pearson相关系数、Spearman秩相关系数
- **ICIR计算**: 信息系数稳定性（均值/标准差）
- **分层回测**: 5/10分组等权测试
- **多空组合**: Top组-Bottom组绩效计算
- **训练/测试集划分**: 按时间切分，避免未来函数
- **行业市值中性化**: 可选回归残差中性化
- **换手率计算**: 因子稳定性评估

### 🎨 可视化报告
- IC时间序列图（带±2σ通道）
- 多空净值曲线图
- 分组累计收益对比图
- 月度IC热力图
- 因子整体对比汇总图表

## 项目结构

```
alpha_mining/
├── main.py                  # 主程序入口
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── data_hub.py             # 统一数据抽象层
├── factor_engine.py        # 因子解析与计算引擎
├── evaluator.py           # 因子评估器
├── visualizer.py          # 结果可视化
├── operators.py           # 算子库
├── utils.py               # 工具函数
│
├── data_adapters/         # 数据源适配器
│   ├── __init__.py
│   ├── base_adapter.py
│   ├── akshare_adapter.py
│   ├── tushare_adapter.py
│   ├── baostock_adapter.py
│   └── yfinance_adapter.py
│
├── data_cache/            # 行情数据缓存（自动生成）
├── factor_cache/          # 因子值缓存（自动生成）
└── results/               # 结果输出（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `config.yaml`:

```yaml
data:
  preferred_source: akshare        # 首选数据源
  start_date: "2020-01-01"        # 数据起始日期
  end_date: "2024-12-31"          # 数据结束日期
  symbols: all_a                    # 股票池

factor:
  expressions:                      # 因子表达式列表
    - "rank(ts_delta(close, 5))"
    - "rank(ts_mean(close, 20) / close)"
    - "-rank(ts_std(return_1d, 20))"
  use_genetic_programming: false    # 是否使用GP自动生成因子

evaluation:
  n_groups: 5                       # 分组数量
  train_ratio: 0.8                 # 训练集比例
  neutralize: false                # 是否中性化

filter:
  min_train_icir: 0.3              # 筛选阈值
  min_test_ic_mean: 0.01
  min_ls_sharpe_train: 0.5
```

### 3. 运行

```bash
# 完整运行（使用缓存）
python main.py

# 强制重新计算（不使用缓存）
python main.py --no-cache

# 不生成图表
python main.py --no-charts

# 指定配置文件
python main.py --config my_config.yaml
```

## 算子参考

### 截面算子（按日期分组计算）

| 算子 | 说明 |
|------|------|
| `rank(x)` | 截面百分位排序 |
| `zscore(x)` | 截面标准化 |
| `demean(x)` | 截面去均值 |
| `scale(x)` | 缩放绝对值和为1 |

### 时序算子（按股票分组滚动计算）

| 算子 | 说明 |
|------|------|
| `ts_mean(x, d)` | d日移动平均 |
| `ts_std(x, d)` | d日移动标准差 |
| `ts_min/ts_max(x, d)` | d日极值 |
| `ts_delta(x, d)` | d日变化量 |
| `ts_pct_change(x, d)` | d日变化率 |
| `shift(x, d)` | 滞后d期 |
| `ts_zscore(x, d)` | 滚动zscore |
| `ts_corr(x, y, d)` | x与y的滚动相关系数 |
| `ts_cov(x, y, d)` | x与y的滚动协方差 |

### 数学与逻辑算子

`abs(x)`, `sign(x)`, `log(x)`, `power(x, a)`, `sqrt(x)`
`add/sub/mul/div(x, y)`, `if_else(cond, x, y)`

## 因子表达式示例

```yaml
# 动量因子
"rank(ts_delta(close, 5))"
"ts_zscore(return_1d, 20)"

# 反转因子
"-rank(ts_mean(return_1d, 5))"
"-rank(ts_mean(return_1d, 20))"

# 波动率因子
"-rank(ts_std(return_1d, 20))"
"rank(high_low_ratio)"

# 成交量因子
"rank(ts_zscore(log_volume, 20))"
"-rank(ts_corr(close, log_volume, 20))"

# 均线因子
"rank(ts_mean(close, 10) / close)"
"rank(ts_mean(close, 5)) - rank(ts_mean(close, 20))"

# 高低位置因子
"rank((close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001))"
```

## 结果解读

### 输出文件

```
results/
├── factor_summary.png          # 因子总体对比图
├── factor_XXX.png              # 单个因子详情图
└── factor_results.csv          # 所有因子评估结果
```

### 关键指标

| 指标 | 说明 | 优秀阈值 |
|------|------|----------|
| RankICIR_train | 训练集信息系数稳定性 | > 0.5 |
| RankIC_mean_test | 测试集IC均值 | > 0.02，方向与训练集一致 |
| 夏普比率_train | 多空组合夏普比率 | > 1.0 |
| 年化收益率_train | 多空组合年化收益 | > 10% |
| 最大回撤_train | 多空组合最大回撤 | < 20% |
| 换手率 | 因子日换手率 | < 0.3 |

## 扩展开发

### 添加新的数据源适配器

1. 继承 `BaseDataAdapter`
2. 实现 `get_daily_data()` 和 `get_industry_classification()`
3. 在 `data_hub.py` 的 `ADAPTER_MAP` 中注册

### 添加新算子

在 `operators.py` 中定义函数，然后加入 `OPERATORS` 字典即可。

### 遗传编程扩展

安装 `gplearn` 并在配置中启用 `use_genetic_programming: true`

## 注意事项

1. **回测存在过拟合风险**: 请务必检查样本外表现，不要过度优化参数
2. **数据源限制**: 免费数据源可能有调用频率限制和数据质量问题
3. **计算性能**: 全市场5年数据+100个因子计算约需5-10分钟
4. **缓存管理**: 定期清理 `data_cache/` 和 `factor_cache/` 目录

## 许可证

MIT License

## 贡献

欢迎提交Issue和PR！
