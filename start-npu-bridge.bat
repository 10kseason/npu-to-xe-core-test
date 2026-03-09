@echo off
setlocal
cd /d "%~dp0"
python -m npu_xmx.cli serve --host 127.0.0.1 --port 8765 --socket-port 8766
