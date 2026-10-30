@echo off
rem GUI を開く。.venv があればそれを、なければ PATH の pythonw を使う
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" name2struct.py
) else (
  start "" pythonw name2struct.py
)
