@echo off
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0copy_source.ps1" -Version v84
pause
