# Vercel Serverless Function 入口
# 重定向到 Streamlit（或使用完整的Streamlit部署）
import json

def handler(request):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "name": "Alpha因子挖掘系统",
            "version": "3.0.0",
            "status": "API Service OK",
            "docs": "Alpha因子挖掘系统 - 一个模块化的自动化Alpha因子挖掘平台",
            "features": [
                "多数据源支持（Akshare/Tushare/Baostock/YFinance）",
                "17个经典量价因子预设",
                "IC/IR因子有效性评估",
                "分层回测分析",
                "可视化报告生成"
            ]
        }, ensure_ascii=False)
    }
