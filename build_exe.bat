@echo off
cd /d %~dp0
python -m PyInstaller --noconfirm --clean --onefile --windowed --icon assets\app_icon.ico --name auto_clicker_cn app.py
pause
