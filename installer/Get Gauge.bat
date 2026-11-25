@echo off
setlocal
title Get Gauge
REM ---------------------------------------------------------------------------------------------
REM  Get Gauge: one file that downloads Gauge and installs everything it needs.
REM  Double-click it. No git, no Python, nothing else required first.
REM
REM  It puts Gauge in  %USERPROFILE%\Gauge  and then runs Gauge's own installer, which installs
REM  Python, Ollama (the local AI engine), Gauge, and the AI models, and puts "Gauge Studio" on
REM  your Desktop. Running it again later updates Gauge and keeps your stories and settings.
REM
REM  Optional, set before running:   set GAUGE_DIR=D:\Gauge      (install somewhere else)
REM                                  set GAUGE_MODEL=qwen2.5:14b (choose the writer model)
REM                                  set GAUGE_ENGINE=native     (no Ollama; built-in model)
REM ---------------------------------------------------------------------------------------------
set "GAUGE_SELF=%~f0"
if not defined GAUGE_BRANCH set "GAUGE_BRANCH=ccr-24dc418f-hsneyx"
if not defined GAUGE_DIR set "GAUGE_DIR=%USERPROFILE%\Gauge"
echo.
echo   Getting Gauge. The first time takes 10-20 minutes: it downloads the AI models (a few GB).
echo   You do not need to do anything. A window called "Gauge Studio" opens when it is done.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:GAUGE_SELF); Invoke-Expression $s.Substring($s.LastIndexOf('#---POWERSHELL---'))"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo   Something did not work. Nothing is broken: you can run this file again.
  echo   If it keeps failing, take a photo of this window and ask for help.
)
echo.
pause
exit /b %RC%
#---POWERSHELL---
$ErrorActionPreference = "Stop"
try {
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
  $ProgressPreference = "SilentlyContinue"
  $repo = "charmand3rrawr-ui/Gauge"
  $dir = $env:GAUGE_DIR
  $branches = @($env:GAUGE_BRANCH, "main") | Select-Object -Unique
  $tmp = Join-Path $env:TEMP ("gauge-kit-" + [Guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $tmp | Out-Null
  $src = $null
  $n = 0
  foreach ($b in $branches) {
    $n++
    $zip = Join-Path $tmp "gauge-$n.zip"
    $out = Join-Path $tmp "unzipped-$n"
    Write-Host "  Downloading Gauge ($b) ..."
    try { Invoke-WebRequest -Uri "https://github.com/$repo/archive/refs/heads/$b.zip" -OutFile $zip -UseBasicParsing }
    catch { Write-Host ("    could not download it: " + $_.Exception.Message); continue }
    Expand-Archive -LiteralPath $zip -DestinationPath $out -Force
    $top = Get-ChildItem -LiteralPath $out -Directory | Select-Object -First 1
    if ($top -and (Test-Path (Join-Path $top.FullName "installer\bootstrap.ps1"))) { $src = $top.FullName; break }
    Write-Host "    that copy has no installer in it; trying the next one"
  }
  if (-not $src) { throw "Could not download Gauge. Check your internet connection and run this file again." }
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  # Copy the program over the install folder. Your data, stories and the Python environment live in
  # folders the download does not contain, so they are left exactly as they were.
  Copy-Item -Path (Join-Path $src "*") -Destination $dir -Recurse -Force
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "  Gauge is in $dir"
  # Remember which version this is, so 'Update Gauge.bat' can tell when there is nothing new.
  try {
    $got = [string](Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits/$b" -TimeoutSec 20).sha
    if ($got) { [IO.File]::WriteAllText((Join-Path $dir "installer\installed-version.txt"), "$b $got") }
  } catch {}
  $boot = Join-Path $dir "installer\bootstrap.ps1"
  $opts = @{}
  if ($env:GAUGE_MODEL) { $opts.Model = $env:GAUGE_MODEL }
  if ($env:GAUGE_ENGINE) { $opts.Engine = $env:GAUGE_ENGINE }
  & $boot @opts
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
  exit 0
} catch {
  Write-Host ""
  Write-Host ("  ERROR: " + $_.Exception.Message) -ForegroundColor Red
  exit 1
}
