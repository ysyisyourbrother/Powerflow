#!/bin/zsh
# Powerflow 开发环境安装
#
# 做两件事:
#   1. 检查仓库自带的 matpower/ 是否完整(它已随仓库提交,无需再下载)
#   2. 在本目录内创建 octave-env/(Octave 解释器)
#
# 不修改 base 或其他 conda 环境,不写任何文件到本目录之外。

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> [1/2] 检查 MATPOWER 源码"
if [ -f "matpower/lib/runpf.m" ]; then
  echo "    matpower/ 已随仓库存在,跳过"
else
  echo "    matpower/ 缺失或损坏。" >&2
  echo "    请重新 clone 本仓库: git clone https://github.com/ysyisyourbrother/Powerflow.git" >&2
  exit 1
fi

echo "==> [2/2] 安装 Octave (conda-forge, osx-arm64, 首次约 340MB 下载)"
if [ -x "octave-env/bin/octave" ]; then
  echo "    octave-env/ 已存在,跳过"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "    找不到 conda。请先安装 miniconda3: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
  fi
  conda create -y -p "$HERE/octave-env" -c conda-forge octave
fi

echo
echo "安装完成。验证环境:"
echo "  ./octave.sh src/examples/smoke_test.m"
