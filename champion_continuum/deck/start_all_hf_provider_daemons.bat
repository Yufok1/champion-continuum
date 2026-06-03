@echo off
setlocal EnableExtensions
REM Champion Continuum - launch the HF Inference Provider free-credit scout pack.
REM Auth comes from HF_TOKEN/HUGGINGFACE_HUB_TOKEN or local `hf auth login`.
REM
REM Usage:
REM   start_all_hf_provider_daemons.bat
REM   start_all_hf_provider_daemons.bat --include-unverified
REM   start_all_hf_provider_daemons.bat --dry-run
REM
REM The default pack starts verified chat routes only. --include-unverified also
REM starts catalog chat-provider attempts that may fail if that provider does
REM not currently serve the default model.

cd /d "%~dp0" || (
  echo   ERROR: could not enter deck root: %~dp0
  exit /b 1
)

if not exist "%~dp0launch_hf_provider_pack.py" (
  echo   ERROR: launch_hf_provider_pack.py is missing beside this batch file.
  exit /b 1
)
if not exist "%~dp0forum_daemon.py" (
  echo   ERROR: forum_daemon.py is missing beside this batch file.
  exit /b 1
)
if not exist "%~dp0forum_hf_agent.py" (
  echo   ERROR: forum_hf_agent.py is missing beside this batch file.
  exit /b 1
)
where python >nul 2>nul || (
  echo   ERROR: python was not found on PATH.
  exit /b 1
)

echo.
echo   Champion Continuum HF provider daemon pack starting...
echo   This starts local forum daemons. Provider credits are used only when a
echo   daemon later answers an assigned turn.
echo.
python "%~dp0launch_hf_provider_pack.py" %*
exit /b %ERRORLEVEL%
