# dataset：算例与比赛数据

本目录存放比赛数据、标准测试系统和预处理产物。

## 当前内容

```text
dataset/
├── ieee39/
│   ├── README.md
│   ├── manifest.json
│   ├── checksums.sha256
│   ├── raw/
│   │   ├── pypower/
│   │   └── pandapower/
│   ├── reference/
│   └── scripts/
└── README.md
```

## IEEE 39 节点系统

已经准备好：

- 39 母线、10 发电机、46 条 PYPOWER 支路；
- 35 条 Pandapower 线路和 11 台变压器；
- 21 个负荷；
- 完整原始 CSV、JSON、NPZ；
- PYPOWER/Pandapower 参考潮流结果；
- 原始数据重新导出脚本和 SHA-256 校验。

入口：

- [IEEE 39 数据说明](ieee39/README.md)
- [PYPOWER 母线表](ieee39/raw/pypower/bus.csv)
- [PYPOWER 发电机表](ieee39/raw/pypower/gen.csv)
- [PYPOWER 支路表](ieee39/raw/pypower/branch.csv)
- [Pandapower 原始 JSON](ieee39/raw/pandapower/case39.json)

## 数据约定

- `raw/` 中的数据保持原始值，不在原文件中直接改写。
- 派生结果放在 `reference/` 或单独的 `processed/`。
- 大文件（超过 10 MB）优先使用 Git LFS 或按需生成。
- 新比赛数据请单独建立子目录，并在对应 `README.md` 中记录字段、单位和来源。
