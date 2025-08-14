@echo off
setlocal
cd /d "%~dp0"

rem Ensure uv is installed and on PATH
where uv >nul 2>&1 || (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

rem Prefer wheels to avoid building lxml from source
set "PIP_ONLY_BINARY=:all:"
set "PIP_PREFER_BINARY=1"

rem Install lxml wheel explicitly on Windows, then run the app
uv run --python 3.11 -m pip install --only-binary=:all: "lxml==5.*"
uv run --python 3.11 -m streamlit run app.py

pause
endlocal
