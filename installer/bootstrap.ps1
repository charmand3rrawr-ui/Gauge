<#
  Gauge zero-questions installer for Windows 10/11. Double-click "Install Gauge.bat" to run it.

  What it does, skipping anything already done:
    1. Python 3.12 (per-user, no admin) if no Python 3.10+ is found
    2. Ollama (the local AI engine) via winget, or a direct download if winget is missing
    3. A private Python environment with Gauge installed into it
    4. The AI models: a writer model, a small fast model for bookkeeping, and an embedding model
    5. Gauge's data folder and settings (fast model and concurrency pre-configured)
    6. A "Gauge Studio" shortcut on the Desktop, then opens it

  Flags (optional): -Model llama3.1:8b  -FastModel llama3.2:3b  -NoLaunch  -NoShortcut
#>
param(
  [string]$Model = "llama3.1:8b",
  [string]$FastModel = "llama3.2:3b",
  [string]$EmbedModel = "nomic-embed-text",
  [switch]$NoLaunch,
  [switch]$NoShortcut
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Log = Join-Path $Root "installer\install.log"
Start-Transcript -Path $Log -Append | Out-Null

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine") }

function Find-Python {
  foreach ($cmd in @("python", "python3", "py")) {
    $c = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($c) {
      try {
        $v = & $c.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -and [version]$v -ge [version]"3.10") { return $c.Source }
      } catch {}
    }
  }
  $local = Join-Path $env:LOCALAPPDATA "Programs\Python"
  if (Test-Path $local) {
    $exe = Get-ChildItem $local -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "venv" } | Sort-Object FullName -Descending | Select-Object -First 1
    if ($exe) { return $exe.FullName }
  }
  return $null
}

function Find-Ollama {
  $c = Get-Command ollama -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
  if (Test-Path $candidate) { return $candidate }
  return $null
}

# ---------------------------------------------------------------- 1. Python
Step "Checking for Python"
$py = Find-Python
if (-not $py) {
  Warn "Python not found. Installing Python 3.12 for this user (no admin needed)..."
  $installed = $false
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try { winget install --id Python.Python.3.12 -e --scope user --silent --accept-source-agreements --accept-package-agreements | Out-Null; $installed = $true } catch { Warn "winget could not install Python: $_" }
  }
  if (-not $installed) {
    $tmp = Join-Path $env:TEMP "python-installer.exe"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $tmp -UseBasicParsing
    Start-Process -FilePath $tmp -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=0" -Wait
  }
  Refresh-Path
  $py = Find-Python
  if (-not $py) { throw "Python installed but not found yet. Close this window, open it again, and run the installer once more." }
}
Ok "Python: $py"

# ---------------------------------------------------------------- 2. Ollama
Step "Checking for Ollama (the local AI engine)"
$ollamaExe = Find-Ollama
if (-not $ollamaExe) {
  Warn "Ollama not found. Installing..."
  $done = $false
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try { winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements | Out-Null; $done = $true } catch { Warn "winget could not install Ollama: $_" }
  }
  if (-not $done) {
    $tmp = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $tmp -UseBasicParsing
    Start-Process -FilePath $tmp -ArgumentList "/S" -Wait
  }
  Refresh-Path
  $ollamaExe = Find-Ollama
  if (-not $ollamaExe) { throw "Ollama did not install. Download it from https://ollama.com/download, install it, and run this again." }
}
Ok "Ollama: $ollamaExe"
foreach ($kv in @(@("OLLAMA_NUM_PARALLEL", "4"), @("OLLAMA_FLASH_ATTENTION", "1"), @("OLLAMA_KV_CACHE_TYPE", "q8_0"))) {
  if (-not [Environment]::GetEnvironmentVariable($kv[0], "User")) { [Environment]::SetEnvironmentVariable($kv[0], $kv[1], "User") }
}
try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
  Warn "Starting the Ollama engine..."
  Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
  $tries = 0
  while ($tries -lt 30) { Start-Sleep -Seconds 2; try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; break } catch { $tries++ } }
}

# ---------------------------------------------------------------- 3. Gauge
Step "Installing Gauge into its own Python environment"
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
& $venvPy -m pip install -e ".[ml]" --quiet --disable-pip-version-check
Ok "Gauge installed"

# ---------------------------------------------------------------- 4. Models
Step "Downloading the AI models (the slow part; a few GB)"
foreach ($m in @($Model, $FastModel, $EmbedModel)) {
  Write-Host "    pulling $m ..."
  & $ollamaExe pull $m
  if ($LASTEXITCODE -ne 0) { Warn "could not pull $m; Gauge still installs. Later, run: ollama pull $m" }
}

# ---------------------------------------------------------------- 5. Settings
Step "Setting up Gauge"
& $venvPy -m gauge.cli init --provider ollama --model $Model | Out-Null
$cfgPath = Join-Path $Root "data\config.json"
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
$cfg.llm.fast_model = $FastModel
$cfg.llm.parallel = 4
$cfg | ConvertTo-Json -Depth 10 | Set-Content $cfgPath -Encoding UTF8
& $venvPy -m gauge.cli doctor
Ok "Settings written to $cfgPath"

# ---------------------------------------------------------------- 6. Shortcut + launch
if (-not $NoShortcut) {
  Step "Creating the Desktop shortcut"
  $desktop = [Environment]::GetFolderPath("Desktop")
  $lnk = Join-Path $desktop "Gauge Studio.lnk"
  $shell = New-Object -ComObject WScript.Shell
  $sc = $shell.CreateShortcut($lnk)
  $sc.TargetPath = Join-Path $Root "Gauge Studio.bat"
  $sc.WorkingDirectory = $Root
  $sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,70"
  $sc.Description = "Write stories with Gauge"
  $sc.Save()
  Ok "Shortcut: $lnk"
}
Stop-Transcript | Out-Null
Write-Host ""
Write-Host "All done. Double-click 'Gauge Studio' on your Desktop any time." -ForegroundColor Green
if (-not $NoLaunch) { Start-Process -FilePath (Join-Path $Root "Gauge Studio.bat") -WorkingDirectory $Root }
