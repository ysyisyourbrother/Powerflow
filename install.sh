#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "错误：未找到 Python 3.11/3.12。" >&2
  exit 1
fi

echo "==> 使用的解释器: $PYTHON"
echo "==> 创建项目虚拟环境 .venv"
"$PYTHON" -m venv .venv

echo "==> 安装 Python 依赖"
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation

echo "==> 运行环境自检"
.venv/bin/python src/examples/smoke_test.py

echo
echo "安装完成。以后运行示例："
echo "  .venv/bin/python src/examples/step1_case9.py"
echo "  .venv/bin/python src/examples/step2_pandapower.py"
