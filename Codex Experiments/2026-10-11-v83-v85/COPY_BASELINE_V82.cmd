@echo off
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0copy_candidate.ps1" -Version v82
pause
