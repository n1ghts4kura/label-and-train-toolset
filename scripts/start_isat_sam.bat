@echo off
chcp 65001 >nul
echo ============================================================
echo  ISAT-SAM Backend 启动器
echo ============================================================
echo.

REM 检查是否在 conda 环境中
where conda >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] 使用 conda 环境
    echo [INFO] 激活 isat_sam 环境...
    call conda activate isat_sam
) else (
    echo [INFO] 使用虚拟环境
    echo [INFO] 激活 venv_isat_sam 环境...
    call venv_isat_sam\Scripts\activate.bat
)

echo.
echo [INFO] 启动 ISAT-SAM Backend...
echo [INFO] 默认地址: http://127.0.0.1:8000
echo [INFO] 按 Ctrl+C 停止服务
echo.

REM 启动 isat-sam-backend
isat-sam-backend --checkpoint mobile_sam.pt --host 127.0.0.1 --port 8000

pause
