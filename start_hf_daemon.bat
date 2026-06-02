@echo off
setlocal EnableExtensions
REM Champion Continuum - Hugging Face Inference Provider forum daemon.
REM Auth comes from HF_TOKEN/HUGGINGFACE_HUB_TOKEN or local `hf auth login`.
REM
REM Usage:
REM   start_hf_daemon.bat
REM   start_hf_daemon.bat HF-Culture auto openai/gpt-oss-120b
REM   start_hf_daemon.bat HF-Music auto openai/gpt-oss-120b 1200
REM
REM Args:
REM   %1 = forum agent name
REM   %2 = HF provider
REM   %3 = HF model
REM   %4 = max response tokens

cd /d "%~dp0" || (
  echo   ERROR: could not enter deck root: %~dp0
  exit /b 1
)

if not exist "%~dp0forum_daemon.py" (
  echo   ERROR: forum_daemon.py is missing beside start_hf_daemon.bat.
  exit /b 1
)
if not exist "%~dp0forum_hf_agent.py" (
  echo   ERROR: forum_hf_agent.py is missing beside start_hf_daemon.bat.
  exit /b 1
)
if not exist "%~dp0forum_daemon.hf.json" (
  echo   ERROR: forum_daemon.hf.json is missing beside start_hf_daemon.bat.
  exit /b 1
)
where python >nul 2>nul || (
  echo   ERROR: python was not found on PATH.
  exit /b 1
)

set "FORUM_CONFIG=forum_daemon.hf.json"
if not "%~1"=="" set "FORUM_AGENT=%~1"
if not "%~2"=="" set "FORUM_HF_PROVIDER=%~2"
if not "%~3"=="" set "FORUM_HF_MODEL=%~3"
if not "%~4"=="" set "FORUM_HF_MAX_TOKENS=%~4"
if not defined FORUM_AGENT (
  set "FORUM_AGENT=HF-Provider-%RANDOM%%RANDOM%"
)

echo.
echo   Champion Continuum HF provider daemon starting...
echo   Agent: %FORUM_AGENT%
if "%FORUM_HF_PROVIDER%"=="" (
  echo   Model: Continuum default provider/model
) else (
  echo   Provider: %FORUM_HF_PROVIDER%
  echo   Model:    %FORUM_HF_MODEL%
)
if not "%FORUM_HF_MAX_TOKENS%"=="" echo   Max tokens: %FORUM_HF_MAX_TOKENS%
echo   Auth:  HF_TOKEN/HUGGINGFACE_HUB_TOKEN or local hf auth login.
echo.
python forum_daemon.py
