@echo off
setlocal

set "LOG=build.log"
echo Iniciando build > %LOG%
echo ==== %DATE% %TIME% ====>> %LOG%

echo Instalando dependencias...
call python -m pip install --upgrade pip >> %LOG% 2>&1
if exist requirements.build.txt (
  call python -m pip install -r requirements.build.txt >> %LOG% 2>&1
) else (
  call python -m pip install -r requirements.txt >> %LOG% 2>&1
)
if errorlevel 1 (
  echo ERROR: fallo la instalacion de dependencias. Revisa %LOG%
  exit /b 1
)
call python -m pip install pyinstaller >> %LOG% 2>&1
if errorlevel 1 (
  echo ERROR: fallo la instalacion de PyInstaller. Revisa %LOG%
  exit /b 1
)

echo.
echo Construyendo exe (esto puede tardar varios minutos)...
echo.
call python -m PyInstaller --clean --noconfirm senxor.spec >> %LOG% 2>&1
if errorlevel 1 (
  echo ERROR: PyInstaller fallo. Revisa %LOG%
  exit /b 1
)

echo.
if exist dist\senxor.exe (
  echo Build listo: dist\senxor.exe
  dir dist\senxor.exe >> %LOG% 2>&1
) else (
  echo ERROR: no se genero dist\senxor.exe
)
if not exist dist\settings.json (
  if exist settings.json (
    copy /y settings.json dist\settings.json >nul
  ) else if exist settings.example.json (
    copy /y settings.example.json dist\settings.json >nul
  )
)
if not exist dist\output mkdir dist\output
echo Configs y output listos en dist\
endlocal
