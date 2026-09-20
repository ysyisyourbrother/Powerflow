#!/bin/zsh
# 启动 Octave + MATPOWER 学习环境
#
# 设计:只依赖 conda 单文件脚本 conda.sh,不改动系统 PATH/conda 配置,
#       不需要 conda activate,不写任何文件到目录之外。
#
# 用法:
#   ./octave.sh                 # 交互式 Octave(自动加载 MATPOWER)
#   ./octave.sh myscript.m      # 运行脚本后退出
#   ./octave.sh --eval "runpf('case9')"

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HERE/octave-env"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [ ! -x "$ENV_DIR/bin/octave" ]; then
  echo "找不到 Octave: $ENV_DIR/bin/octave" >&2
  echo "请先运行 ./install.sh" >&2
  exit 1
fi
if [ ! -f "$CONDA_SH" ]; then
  echo "找不到 conda 初始化脚本: $CONDA_SH" >&2
  echo "请确认 miniconda3 安装路径,或改用: conda run -p $ENV_DIR octave" >&2
  exit 1
fi

# 加载 conda 函数并激活本地前缀环境(激活只影响当前 shell 进程)
source "$CONDA_SH"
conda activate "$ENV_DIR"

# 切到 workspace,Octave 会自动读取同目录的 startup.m(其中 addpath MATPOWER)
cd "$HERE"

exec octave --no-gui --quiet "$@"
