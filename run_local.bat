@echo off
chcp 65001 >nul
title A股本地选股

REM ============================================================
REM  一键：本地跑全市场分析 → 自动打开网页看结果
REM  用法：双击本文件即可（首次请先看同目录 本地运行说明）
REM ============================================================

REM 清掉“强制全局代理”的环境变量，让国内行情接口直连
REM （Clash/代理软件自己的规则分流不受影响）
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

REM 中长线分析范围：0 = 全市场（慢但完整）；想快速测试可改成 800
set LONG_TERM_MAX_UNIVERSE=0

cd /d "%~dp0scripts"

REM 先检查 tinyshare 是否已安装
python -c "import tinyshare" 2>nul
if errorlevel 1 (
  echo.
  echo [缺少 tinyshare] 请先在命令行执行以下命令安装（需代理可用）：
  echo   pip install tinyshare --extra-index-url https://minidoc.pages.dev/simple/
  echo.
  pause
  exit /b 1
)

echo ==========================================
echo   开始分析全市场 A 股，请耐心等待...
echo   （分析期间不要关闭本窗口）
echo ==========================================
echo.
python run_selectors.py

echo.
echo ==========================================
echo   分析完成！正在打开网页...
echo   浏览器打开后，看完直接关闭本窗口即可
echo   （若提示无法连接，稍等一两秒刷新一下）
echo ==========================================
cd site
start "" http://localhost:8765/
python -m http.server 8765
