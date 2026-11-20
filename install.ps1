<#
.SYNOPSIS
  One-shot Windows installer for Gauge.
.DESCRIPTION
  Creates a virtual environment, installs Gauge, checks for Ollama, pulls the default models,
  and initialises the data directory. Safe to re-run.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\install.ps1
  powershell -ExecutionPolicy Bypass -File .\install.ps1 -Model "qwen2.5:14b" -WithML
#>
param(
  [string]$Model = "llama3.1:8b",
  [string]$EmbedModel = "nomic-embed-text",
  [switch]$WithML,          # also install scikit-learn for the stronger engagement model
  [switch]$SkipOllama       # don't try to install/pull Ollama models
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking Python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) {
  Write-Host "Python 3.10+ not found. Install from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH') or run: winget install Python.Python.3.12" -ForegroundColor Yellow
  exit 1
}
$ver = & $py.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "Python $ver at $($py.Source)"
if ([version]$ver -lt [version]"3.10") { Write-Host "Python 3.10+ required" -ForegroundColor Red; exit 1 }

Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv")) { & $py.Source -m venv .venv }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Step "Installing Gauge"
& $venvPy -m pip install --upgrade pip --quiet
if ($WithML) { & $venvPy -m pip install -e ".[ml,dev]" --quiet } else { & $venvPy -m pip install -e ".[dev]" --quiet }

if (-not $SkipOllama) {
  Step "Checking Ollama"
  $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  if (-not $ollama) {
    Write-Host "Ollama not found. Installing with winget (or download from https://ollama.com/download)..." -ForegroundColor Yellow
    try { winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements } catch { Write-Host "winget failed; install Ollama manually from https://ollama.com/download then re-run." -ForegroundColor Yellow }
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  }
  if ($ollama) {
    Step "Pulling models ($Model, $EmbedModel) - this downloads a few GB the first time"
    try { ollama pull $Model } catch { Write-Host "Could not pull $Model (is the Ollama service running?)" -ForegroundColor Yellow }
    try { ollama pull $EmbedModel } catch { Write-Host "Could not pull $EmbedModel" -ForegroundColor Yellow }
  }
}

Step "Initialising data directory"
& $venvPy -m gauge.cli init --provider ollama --model $Model
& $venvPy -m gauge.cli doctor

Write-Host "`nDone. Use .\gauge.bat from this folder, e.g.:" -ForegroundColor Green
Write-Host "  .\gauge.bat discover royalroad best-rated --pages 2"
Write-Host "  .\gauge.bat ingest"
Write-Host "  .\gauge.bat train"
Write-Host "  .\gauge.bat new ""My Serial"" --idea ""a courier who can rewind ten seconds"" --genre Sci-fi"
Write-Host "  .\gauge.bat write ""My Serial"" --chapters 3"
Write-Host "  .\gauge.bat daemon --project ""My Serial"" --write"
