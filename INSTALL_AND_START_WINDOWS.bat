@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (
  echo Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements-portable.txt
)
start "" http://127.0.0.1:4173
.venv\Scripts\python.exe local_server.py
pause
