@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run this first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py

if errorlevel 1 (
    pause
)
