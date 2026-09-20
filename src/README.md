# src/ —— 算法入口代码

比赛算法的入口就放在这里。

规划建议(等你拓展时再落):

```
src/
├── main.m            # 主入口: 读 dataset/ -> 跑算法 -> 输出结果
├── solve.m           # 算法核心(可在这里调用/替换 matpower/lib 下的潮流求解器)
├── preprocess.m      # 数据预处理
└── utils/            # 辅助函数
```

运行方式(环境会自动把本目录加进搜索路径):

```bash
cd ~/CodeSpace/Powerflow
./octave.sh src/main.m
```

现在目录是空的,等你开始写算法时填入。

写代码时的两个约定见顶层 `README.md`:
- 想**覆盖** MATPOWER 官方函数(例如换成你自己的牛顿法),在 `src/` 里放同名 `.m` 文件,
  它会优先于 `matpower/lib/` 里的官方版本被加载。
- 只想**新增**函数,直接写即可,不会和官方冲突。
