@echo off
setlocal EnableExtensions
REM Champion Continuum - one-click local deck launcher.
REM Double-click this, or run: start_deck.bat
set "DRY_RUN="
if /I "%~1"=="--dry-run" set "DRY_RUN=1"
if /I "%~1"=="/dry-run" set "DRY_RUN=1"
if /I "%~1"=="dry-run" set "DRY_RUN=1"
set "CLEAN_DRY_RUN="
if defined DRY_RUN set "CLEAN_DRY_RUN=-DryRun"

cd /d "%~dp0" || (
  echo   ERROR: could not enter deck root: %~dp0
  exit /b 1
)
set "CONTINUUM_CLI_BRAIN=1"
set "CONTINUUM_BRAIN_CHANNEL=%~dp0cli_brain_channel"
set "GRADIO_SERVER_PORT=7870"
set "GRADIO_SERVER_NAME=127.0.0.1"
set "CONTINUUM_LINK_PORT=7871"
set "CONTINUUM_LINK_URL=http://127.0.0.1:%CONTINUUM_LINK_PORT%"
set "CONTINUUM_MCP_PORT=7872"
set "CONTINUUM_MCP_URL=http://127.0.0.1:%CONTINUUM_MCP_PORT%"
set "CONTINUUM_LINK_TOKEN_FILE=%~dp0cli_brain_channel\continuum_link_token.txt"

if not exist "%~dp0app.py" (
  echo   ERROR: app.py is missing beside start_deck.bat.
  exit /b 1
)
if not exist "%~dp0continuum_link_server.py" (
  echo   ERROR: continuum_link_server.py is missing beside start_deck.bat.
  exit /b 1
)
if not exist "%~dp0continuum_mcp_server.py" (
  echo   ERROR: continuum_mcp_server.py is missing beside start_deck.bat.
  exit /b 1
)
if not exist "%~dp0start_deck_clean.ps1" (
  echo   ERROR: start_deck_clean.ps1 is missing beside start_deck.bat.
  exit /b 1
)
where python >nul 2>nul || (
  echo   ERROR: python was not found on PATH.
  exit /b 1
)

echo.
echo   Champion Continuum clean slate...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_deck_clean.ps1" -Port %GRADIO_SERVER_PORT% -LinkPort %CONTINUUM_LINK_PORT% -McpPort %CONTINUUM_MCP_PORT% %CLEAN_DRY_RUN%
if errorlevel 1 (
  echo   Cleanup reported a warning; continuing with deck launch.
)

if not exist "%CONTINUUM_BRAIN_CHANNEL%" mkdir "%CONTINUUM_BRAIN_CHANNEL%" || (
  echo   ERROR: could not create cli_brain_channel.
  exit /b 1
)
if defined DRY_RUN (
  if exist "%CONTINUUM_LINK_TOKEN_FILE%" (
    for /f "usebackq delims=" %%T in ("%CONTINUUM_LINK_TOKEN_FILE%") do set "CONTINUUM_LINK_TOKEN=%%T"
  ) else (
    set "CONTINUUM_LINK_TOKEN=dry-run-token-not-created"
  )
) else (
  if not exist "%CONTINUUM_LINK_TOKEN_FILE%" (
    echo   Creating local Continuum link token...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $dir=Split-Path -Parent $env:CONTINUUM_LINK_TOKEN_FILE; New-Item -ItemType Directory -Force -Path $dir | Out-Null; $b=New-Object byte[] 32; $rng=[Security.Cryptography.RandomNumberGenerator]::Create(); try { $rng.GetBytes($b) } finally { $rng.Dispose() }; $token=[Convert]::ToBase64String($b).TrimEnd('=').Replace('+','-').Replace('/','_'); Set-Content -LiteralPath $env:CONTINUUM_LINK_TOKEN_FILE -NoNewline -Encoding ascii -Value $token"
    if errorlevel 1 (
      echo   ERROR: failed to create Continuum link token.
      exit /b 1
    )
  )
  for /f "usebackq delims=" %%T in ("%CONTINUUM_LINK_TOKEN_FILE%") do set "CONTINUUM_LINK_TOKEN=%%T"
)
if not defined CONTINUUM_LINK_TOKEN (
  echo   ERROR: Continuum link token file was empty.
  exit /b 1
)

if defined DRY_RUN (
  echo.
  echo   Dry run complete. Launch order is ready:
  echo   1. Clean stale deck/link/MCP/forum processes on ports %GRADIO_SERVER_PORT%, %CONTINUUM_LINK_PORT%, and %CONTINUUM_MCP_PORT%.
  echo   2. Keep shared_store intact and create/read local link token.
  echo   3. Start link service: python continuum_link_server.py --port %CONTINUUM_LINK_PORT%.
  echo   4. Start MCP service: python continuum_mcp_server.py --port %CONTINUUM_MCP_PORT%.
  echo   5. Start deck: python app.py.
  echo.
  echo   Deck:  http://127.0.0.1:%GRADIO_SERVER_PORT%
  echo   Link:  %CONTINUUM_LINK_URL%
  echo   MCP:   %CONTINUUM_MCP_URL%/mcp/sse
  echo   Token: cli_brain_channel\continuum_link_token.txt
  exit /b 0
)

echo.
echo   Champion Continuum link service starting...
echo   Link:  %CONTINUUM_LINK_URL%/sse?slot=personal
echo   Token: stored locally at cli_brain_channel\continuum_link_token.txt
start "Champion Continuum Link" /b python "%~dp0continuum_link_server.py" --port %CONTINUUM_LINK_PORT%
echo.
echo   Champion Continuum MCP service starting...
echo   MCP:   %CONTINUUM_MCP_URL%/mcp/sse
start "Champion Continuum MCP" /b python "%~dp0continuum_mcp_server.py" --port %CONTINUUM_MCP_PORT%
echo.
echo   Champion Continuum deck starting...
echo   Open:  http://127.0.0.1:7870
echo   Link:  %CONTINUUM_LINK_URL%
echo   MCP:   %CONTINUUM_MCP_URL%/mcp/sse
echo   (Then open the "Connect an agent" panel and paste that code to your AI.)
echo.
python app.py
