# dataset：算例、比赛数据与数据集文档

本目录存放比赛数据、标准测试系统和预处理产物，以及围绕这些数据的设计说明。
数据和文档放在一起，避免"文档目录说一套、数据目录放一套"。

## 当前内容

```text
dataset/
├── README.md                      # 本文件：数据与文档总入口
├── competition-dataset.md         # 比赛官方字段表：节点表和支路表
├── competition-dataset-guide.md   # 完整教程：从 IEEE 39 算例到可训练样本
├── model-design.md                # 下一册：神经网络输入、架构与训练设计
├── case39_network.png             # IEEE 39 网架全景图
├── figures/
│   ├── case39_features.png        # 阶段 0 特征冗余分析
│   └── case39_sampling.png        # 2000 个断面的采样诊断
└── ieee39/
    ├── README.md
    ├── manifest.json
    ├── checksums.sha256
    ├── raw/
    │   └── pypower/
    ├── reference/
    ├── processed/
    │   └── competition_v1/        # 赛事对齐格式（当前主用）
    └── scripts/
```

## 文档导航

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [competition-dataset.md](competition-dataset.md) | **比赛官方字段表**：节点表和支路表的原始列名、含义和示例 | 查比赛给了什么字段时 |
| [competition-dataset-guide.md](competition-dataset-guide.md) | **数据集教程**：从 IEEE 39 算例讲到字段、掩码、打标、读取和训练建议 | 想系统理解数据时 |
| [model-design.md](model-design.md) | **模型设计**：v1 复盘、基线对比、输入/输出/损失/数据划分的 v2 方案（v1 已实现，v2 待实现） | 准备训练模型时 |
| [ieee39/processed/competition_v1/README.md](ieee39/processed/competition_v1/README.md) | **数据集字段手册**：字段表、单位、掩码语义、统计量、生成命令、已知局限 | 查字段时 |
| [ieee39/README.md](ieee39/README.md) | IEEE 39 原始数据、参考潮流结果、重新导出脚本 | 查原始字段时 |
| [../doc/case39/README.md](../doc/case39/README.md) | IEEE 39 系统 10 章教程：母线类型、牛顿法、结果解读 | 补电力系统背景时 |
| [../doc/README.md](../doc/README.md) | 项目全部文档索引 | 找不到东西时 |

## IEEE 39 节点系统

![IEEE 39 网架全景](case39_network.png)

更多分析图：

- [阶段 0 特征冗余分析](figures/case39_features.png)：当时节点特征多达 12 列；
  当前 `competition_v1` 已收敛为 `[P, Q, V]` 三列。
- [2000 个断面的采样诊断](figures/case39_sampling.png)：σ 分布、平衡机出力、
  网损、逐母线扰动、电容器例外、相角整体平移。

已经准备好：

- 39 母线、10 发电机、46 条 PYPOWER 支路（34 条线路 + 12 条变压器等值支路）；
- 21 个负荷；
- 完整原始 CSV、JSON、NPZ；
- PYPOWER 参考潮流结果；
- 原始数据重新导出脚本和 SHA-256 校验；
- **赛事对齐数据集 `competition_v1`**：2000 条样本，输入是带掩码的
  节点净注入 P/Q 和电压幅值 V，输出是全部节点 Vm 和全部支路潮流。

入口：

- [完整数据集教程](competition-dataset-guide.md)
- [IEEE 39 数据说明](ieee39/README.md)
- [赛事对齐数据集说明](ieee39/processed/competition_v1/README.md)
- [PYPOWER 母线表](ieee39/raw/pypower/bus.csv)
- [PYPOWER 发电机表](ieee39/raw/pypower/gen.csv)
- [PYPOWER 支路表](ieee39/raw/pypower/branch.csv)

## processed 数据

`processed/` 目前只保留一套主用数据：

| 目录 | 用途 | 输入 | 输出 |
|---|---|---|---|
| `competition_v1/` | 对齐比赛格式，当前比赛场景 | 掩码后的节点 P/Q/V | 全部节点 Vm + 全部支路 Δθ/Pf/Pt/Qf/Qt |

`competition_v1` 对应现阶段的比赛场景：发电计划、电压设定和网架都固定，
只扰动负荷。共 2000 条样本，分成 4 个分片。

后续如果加入拓扑扰动、发电调度等新场景，另建子目录，并在对应 README 中
记录字段、单位和生成方式。

## 数据约定

- `raw/` 中的数据保持原始值，不在原文件中直接改写。
- 派生结果放在 `reference/` 或单独的 `processed/`。
- 大文件（超过 10 MB）优先使用 Git LFS 或按需生成。
- 新比赛数据请单独建立子目录，并在对应 `README.md` 中记录字段、单位和来源。
