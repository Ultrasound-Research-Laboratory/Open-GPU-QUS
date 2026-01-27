@echo off
set ROOT=%~dp0..
set OUT=%ROOT%\dist
set REL=%OUT%\GPU-QUS-GUI-Windows-GPU

echo Building release in %REL%

if exist "%REL%" rmdir /s /q "%REL%"
mkdir "%REL%"

xcopy "%ROOT%\apps" "%REL%\apps" /E /I /Y
xcopy "%ROOT%\release_template" "%REL%\release" /E /I /Y
copy "%ROOT%\requirements_gpu.txt" "%REL%\requirements.txt"
copy "%ROOT%\release_template\README_release.txt" "%REL%\README.txt"

cd /d "%OUT%"
powershell -Command "Compress-Archive -Force '%REL%\*' 'GPU-QUS-GUI-Windows-GPU.zip'"

echo Release ZIP created:
echo %OUT%\GPU-QUS-GUI-Windows-GPU.zip
pause
