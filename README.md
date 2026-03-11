文件结构设计：
Drought_Prediction_LSTM/
│
├── data/                          # 数据存放区 
│   ├── raw/                       # 【千万不要改动】这里放你刚下载的原始CSV
│   │   ├── MOD13Q1_NDVI.csv       # NASA AppEEARS 下载的
│   │   ├── NOAA_Weather.csv       # NOAA 下载的气象数据
│   │   └── dataset_description.md # [必写] 数据集描述文档(作业要求)
│   │
│   └── processed/                 # 【程序生成】清洗和对齐后的数据
│       └── merged_data.csv        # 融合后的总表(可以直接喂给模型)
│
├── src/                           # 源代码区 (模块化设计) 
│   ├── __init__.py
│   ├── config.py                  # 存放超参数 (如: 学习率, 窗口大小)
│   ├── data_loader.py             # 负责读取CSV、插值、滑窗处理
│   ├── model.py                   # 定义 LSTM 模型结构
│   ├── train.py                   # 训练循环、Loss计算、WandB记录
│   ├── evaluate.py                # 预测未来、计算 RMSE/MAE
│   └── utils.py                   # 绘图函数、辅助工具
│
├── tests/                         # 单元测试区 
│   ├── __init__.py
│   ├── test_data_loader.py        # 测试数据加载有没有报错
│   └── test_model.py              # 测试模型输入输出维度对不对
│
├── notebooks/                     # 实验与草稿区 [cite: 52]
│   ├── 01_Data_Exploration.ipynb  # 这里的图可以截图放到报告里
│   └── 02_Training_Demo.ipynb     # 用于展示给助教看的完整流程
│
├── results/                       # 结果保存区
│   ├── models/                    # 保存训练好的 .pth 模型文件
│   └── figures/                   # 保存生成的 loss 曲线、预测对比图
│
├── requirements.txt               # 依赖库列表 (pip install -r ...)
├── README.md                      # 项目说明书 (Github首页内容)
└── report.pdf                     # 最终的 MCM 格式报告