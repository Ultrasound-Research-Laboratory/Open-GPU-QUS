@echo off
setlocal
set HERE=%~dp0

call "%HERE%env\Scripts\activate.bat"

if exist "%HERE%env\Scripts\conda-unpack.exe" (
  "%HERE%env\Scripts\conda-unpack.exe" >nul 2>&1
)

cd /d "%HERE%apps\bsc"
python ui_app_bsc.py

pause
endlocal
