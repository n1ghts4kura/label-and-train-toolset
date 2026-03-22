@echo off
chcp 65001 >nul
echo ============================================================
echo  Label Toolkit 环境初始化
echo ============================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python 已安装
python --version

REM 创建虚拟环境（如果不存在）
if not exist "venv" (
    echo.
    echo [INFO] 创建虚拟环境 venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
echo.
echo [INFO] 激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级 pip
echo.
echo [INFO] 升级 pip...
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo [INFO] 安装 Python 依赖...
pip install pyyaml pillow numpy opencv-python

REM 安装 ONNX Runtime
pip install onnxruntime

REM 安装 onnxslim（模型简化）
pip install onnxslim

REM 克隆 yolov5（如果不存在）
if not exist "yolov5" (
    echo.
    echo [INFO] 克隆 YOLOv5 仓库...
    git clone https://github.com/ultralytics/yolov5.git yolov5
    if errorlevel 1 (
        echo [WARN] git clone 失败，请手动克隆：
        echo   git clone https://github.com/ultralytics/yolov5.git yolov5
    ) else (
        echo [INFO] YOLOv5 克隆完成
    )
) else (
    echo.
    echo [INFO] YOLOv5 目录已存在，跳过克隆
)

REM 创建必要目录
echo.
echo [INFO] 创建必要目录...
if not exist "origin_pics" mkdir origin_pics
if not exist "train_data" mkdir train_data

REM 检查 CUDA
echo.
echo [INFO] 检查 CUDA...
python -c "import onnxruntime as ort; print(f'    ONNX Runtime 提供商: {ort.get_available_providers()}')"

echo.
echo ============================================================
echo  环境初始化完成！
echo ============================================================
echo.
echo 接下来可以:
echo   1. 将 ISAT 标注文件放到 origin_pics/
echo   2. 创建 configs/projects/your_project.yaml 配置文件
echo   3. 运行 python scripts/export_labels.py
echo.
pause
