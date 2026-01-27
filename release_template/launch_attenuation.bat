@echo off
setlocal
set HERE=%~dp0

REM Activate packaged environment
call "%HERE%env\Scripts\activate.bat"

REM Fix conda paths (safe to run multiple times)
if exist "%HERE%env\Scripts\conda-unpack.exe" (
  "%HERE%env\Scripts\conda-unpack.exe" >nul 2>&1
)

cd /d "%HERE%apps\attenuation"
python ui_app_paper.py

pause
endlocal
