# Powerflow

电力系统潮流计算 —— 比赛项目。

基于 **MATPOWER 8.1** 框架,用 **Octave** 运行(无需 MATLAB,无需编译)。

---

## 目录结构

```
Powerflow/
├── src/            # 算法入口代码 ← 写算法的地方
│   ├── README.md
│   └── examples/
│       └── smoke_test.m    # 环境自检 / 冒烟测试
├── dataset/        # 比赛数据集 ← 放比赛方的输入数据
├── matpower/       # MATPOWER 8.1 源码(纯 .m 文件,已内嵌到本仓库)
├── octave-env/     # Octave 解释器(本地,不提交,见 .gitignore)
├── startup.m       # Octave 启动时自动加载 MATPOWER + src/
├── octave.sh       # 启动入口
├── install.sh      # 重建开发环境(队友 clone 后跑这个)
└── README.md
```

---

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/ysyisyourbrother/Powerflow.git
cd Powerflow
```

### 2. 建环境(只需一次)

`octave-env/` 没有提交到仓库(体积大、含绝对路径),用脚本重建:

```bash
./install.sh
```

依赖:本机装有 [miniconda3](https://docs.conda.io/en/latest/miniconda.html)(或 Anaconda)。
脚本只在**本目录内**创建 `octave-env/`,不动 conda 配置,不写任何文件到目录外。

### 3. 跑自检

```bash
./octave.sh src/examples/smoke_test.m
```

看到 `[PASS] 环境正常` 就说明装好了。

---

## 日常使用

```bash
cd Powerflow
./octave.sh                    # 交互式 Octave(已自动加载 MATPOWER)
./octave.sh src/main.m         # 跑算法入口
```

进入 Octave 后:

```matlab
runpf('case9')       % 9 节点牛顿法潮流
runpf('case14')
runopf('case30')     % 最优潮流
help runpf
help mpoption        % 所有选项(算法/容差/迭代上限)
```

---

## 写算法

代码放在 `src/`。`startup.m` 会把 `src/` 用 `addpath(..., '-begin')` 放到搜索路径**最前面**,所以:

- **想覆盖** MATPOWER 官方函数(例如换成你自己的牛顿法):在 `src/` 里放同名 `.m` 文件即可,
  它会优先于 `matpower/lib/` 的官方版本被加载,官方源码保持原样。
- **想新增**函数:直接写,不会冲突。

想读潮流计算的主干,按这个顺序看 `matpower/lib/`:

| 文件 | 作用 |
|---|---|
| `runpf.m` | 主流程:读算例 → 内部编号转换 → 选算法 → 求解 → 输出 |
| `newtonpf.m` | 牛顿法迭代主体 |
| `dSbus_dV.m` | 雅可比矩阵组装 |
| `makeYbus.m` | 节点导纳矩阵 Ybus |
| `pfsoln.m` | 由解回算发电机/负荷功率 |
| `ext2int.m` | 外部编号 ↔ 内部编号(读潮流代码前必看) |

---

## 数据

比赛数据集放 `dataset/`。当前为空,等比赛方提供后填入;若数据是超大文件,
建议加进 `.gitignore` 或改用 Git LFS,避免仓库过重。

---

## 环境说明

| 项 | 版本 |
|---|---|
| Octave | 10.3.0(conda-forge,osx-arm64 原生,含绘图) |
| MATPOWER | 8.1(2025-07 发布,要求 Octave ≥ 6.2 / MATLAB ≥ R2016a) |

**不需要编译。** MATPOWER 是纯 `.m` 文件,官方的 `install_matpower` 只是往搜索路径加目录 ——
本项目用 `startup.m` 自动完成。

MATPOWER 文档见 `matpower/docs/MATPOWER-manual.pdf`,官网 https://matpower.org

---

## 三个已知的坑

1. **不要用 `genpath('matpower')` 加载。** 仓库里有 785 个重名 `.m` 文件(测试目录 `*/lib/t/`
   与各子模块之间),全量加载会让 `which()` 找到错误版本,`test_matpower` 直接失败。
   `startup.m` 已改成官方 `install_matpower` 的精确目录清单。

2. **不要直接跑 `octave-env/bin/octave`。** Octave 启动器运行时靠 `OCTAVE_HOME=$CONDA_PREFIX`
   定位真正的二进制,该变量由 conda 的 activate 脚本设置。**一律用 `./octave.sh` 启动。**

3. 算法选项名是 `'NR'` / `'FDXB'` / `'GS'`,**没有** `'FDPF'` 这个名字。

---

## 卸载环境

```bash
rm -rf octave-env          # 只删环境,代码保留
```

彻底清理 conda 共享缓存(可选,只删缓存不影响已装环境):

```bash
conda clean --all
```
