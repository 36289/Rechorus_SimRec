# SimRec

基于 ReChorus 框架的 SimRec 推荐模型实现。

## 项目概述

SimRec 是一个基于知识蒸馏的推荐模型，使用 GNN 作为教师模型，MLP 作为学生模型。本实现基于 ReChorus 框架，实现了模型的训练和评估。

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- ReChorus 框架

## 项目结构

### Source Code

`main.py` serves as the entrance of our framework, and there are three main packages.

### Structure

- `helpers\`
  - `BaseReader.py`: read dataset csv into DataFrame and append necessary information
  - `ContextReader.py`: inherited from BaseReader
  - `ContextSeqReader.py`: inherited from ContextReader
  - `ImpressionReader.py`: inherited from BaseReader
  - `BaseRunner.py`: control the training and evaluation process
  - `CTRRunner.py`: inherited from BaseRunner
  - `ImpressionRunner.py`: inherited from BaseRunner
- `models\`
  - `BaseModel.py`: basic model classes and dataset classes
  - `BaseContextModel.py`: inherited from BaseModel
  - `BaseImpressionModel.py`: inherited from BaseModel
  - `SimRec.py`: our implementation
- `runners\`
  - `__init__.py`:
  - `SimRecRunner.py`: inherited from BaseRunner
- `utils\`
  - `layers.py`: common modules for model definition
  - `utils.py`: some utils functions
  - `plot_metrics_comparison.py`: visualization tools
- `main.py`: main entrance
- `exp.py`: repeat experiments

## 模型架构

- 教师模型：基于 LightGCN 的图神经网络
- 学生模型：多层感知机 (MLP)
- 知识蒸馏：包含预测层面和嵌入层面的蒸馏

## 主要特点

1. ReChorus 框架集成

   - 继承 BaseModel 和 BaseRunner
   - 兼容框架的数据处理流程
   - 支持框架的评估指标

2. 双层蒸馏机制

   - 预测层面的知识蒸馏
   - 嵌入层面的知识蒸馏

3. 可配置性
   - 支持通过配置文件调整参数
   - 灵活的模型架构设置

## 评估指标

- Recall@K
- NDCG@K
- 预测分数分布

## 引用

如果您使用了本代码，请引用：

```
@inproceedings{simrec,
  title={SimRec: 基于知识蒸馏的简单高效推荐模型},

}
```

## 许可证

本项目采用 MIT 许可证
