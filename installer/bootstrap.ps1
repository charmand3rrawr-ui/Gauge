<#
  Gauge zero-questions installer for Windows 10/11. Double-click "Install Gauge.bat" to run it
  (or "Get Gauge.bat", which downloads Gauge first and then runs this).

  What it does, skipping anything already done:
    1. Looks at this computer (memory, graphics card, free disk) and picks a writer model that fits
    2. Python 3.12 (per-user, no admin) if no Python 3.10+ is found
    3. Ollama (the local AI engine) via winget, or a direct download if winget is missing
    4. A private Python environment with Gauge installed into it
    5. The AI models: a writer model, a small fast model for bookkeeping, and an embedding model
    6. Gauge's data folder and settings
    7. Asks the AI a real test question, prints a checklist of what worked, puts a "Gauge Studio"
       shortcut on the Desktop, then opens it

  Safe to run again: it updates Gauge and keeps your stories and settings. Existing settings are only
  changed when you name a model or engine yourself.

  Flags (all optional):
    -Model qwen2.5:14b     choose the writer model yourself instead of letting the installer pick
    -FastModel llama3.2:3b the small helper model
    -Engine native         skip Ollama and use GaugeLM, Gauge's built-in model (you train it yourself)
    -NoLaunch  -NoShortcut

  Written for Windows PowerShell 5.1, the version every Windows 10/11 has.
#>
param(
  [string]$Model = "",
  [string]$FastModel = "llama3.2:3b",
  [string]$EmbedModel = "nomic-embed-text",
  [ValidateSet("ollama", "native")][string]$Engine = "ollama",
  [switch]$NoLaunch,
  [switch]$NoShortcut
)
$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Log = Join-Path $Root "installer\install.log"
try { Start-Transcript -Path $Log -Append | Out-Null } catch {}

$Checks = New-Object System.Collections.ArrayList
function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Check($name, $passed, $note, [switch]$Optional) {
  [void]$Checks.Add([pscustomobject]@{ Name = $name; Passed = [bool]$passed; Note = $note; Optional = [bool]$Optional })
}
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

function Get-RamGB {
  try { return [int][math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { return 0 }
}

function Get-VramGB {
  # Only NVIDIA reports its memory reliably. Other cards are treated as "no GPU" and sized by RAM.
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($smi) {
    try {
      $mb = & $smi.Source --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
      if ($mb) { return [int][math]::Round([double]$mb / 1024) }
    } catch {}
  }
  return 0
}

function Get-FreeGB($path) {
  try {
    $letter = (Resolve-Path $path).Path.Substring(0, 1)
    return [int][math]::Floor((Get-PSDrive -Name $letter).Free / 1GB)
  } catch { return -1 }
}

# Approximate download sizes in GB, for the disk check.
$ModelSizeGB = @{ "llama3.2:3b" = 2; "llama3.1:8b" = 5; "qwen2.5:7b" = 5; "qwen2.5:14b" = 9; "nomic-embed-text" = 1 }
function Size-Of($m) { if ($ModelSizeGB.ContainsKey($m)) { return $ModelSizeGB[$m] } else { return 6 } }

# ---------------------------------------------------------------- 1. This computer
Step "Looking at this computer"
$ram = Get-RamGB
$vram = Get-VramGB
$modelsDir = $env:OLLAMA_MODELS
if (-not $modelsDir) { $modelsDir = $env:USERPROFILE }
$free = Get-FreeGB $modelsDir
if ($vram -gt 0) { $gpuText = "NVIDIA graphics card with $vram GB" } else { $gpuText = "no NVIDIA graphics card found (the AI will run on the processor: slower, still works)" }
if ($free -ge 0) { $freeText = "$free GB" } else { $freeText = "unknown" }
Ok "Memory: $ram GB.  Graphics: $gpuText.  Free disk: $freeText."

# An existing config.json is yours: unless you named a model or engine yourself, its choices are kept,
# so running this again never resets them (and only downloads what is missing).
$ConfigPath = Join-Path $Root "data\config.json"
$Explicit = $PSBoundParameters.ContainsKey("Model") -or $PSBoundParameters.ContainsKey("FastModel") -or $PSBoundParameters.ContainsKey("Engine")
$Eng = $Engine          # a separate variable: the parameter only accepts "ollama" or "native"
$Keep = $false
if ((Test-Path $ConfigPath) -and -not $Explicit) {
  try {
    $saved = (Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json).llm
    if ($saved -and $saved.provider) {
      $Keep = $true
      if ($saved.provider -eq "ollama") {
        $Eng = "ollama"; $Model = $saved.model
        if ($saved.fast_model) { $FastModel = $saved.fast_model } else { $FastModel = $saved.model }
      } elseif ($saved.provider -eq "native") { $Eng = "native" } else { $Eng = "other" }
      Ok "Keeping your existing settings (engine: $($saved.provider), writer model: $($saved.model))"
    }
  } catch { Warn "Could not read the existing settings; they will be written again." }
}
if ($Eng -eq "ollama" -and -not $Keep) {
  if (-not $Model) {
    if ($vram -ge 12) { $Model = "qwen2.5:14b" }
    elseif ($vram -ge 6) { $Model = "llama3.1:8b" }
    elseif ($ram -ge 16) { $Model = "llama3.1:8b" }
    else { $Model = "llama3.2:3b" }
    Ok "Chosen writer model for this computer: $Model"
  } else {
    Ok "Writer model (your choice): $Model"
  }
  $need = (Size-Of $Model) + 4            # the model, plus Ollama, Python and Gauge
  if ($FastModel -ne $Model) { $need += (Size-Of $FastModel) }
  if ($free -ge 0 -and $free -lt $need -and $Model -ne "llama3.2:3b") {
    Warn "Only $free GB free; $Model needs about $need GB. Using the small model (llama3.2:3b) instead."
    $Model = "llama3.2:3b"
  }
  if ($Model -eq "llama3.2:3b") { $FastModel = $Model }   # one small model does both jobs: one download
  if ($free -ge 0 -and $free -lt 6) {
    throw "Not enough free disk space: $free GB free, at least 6 GB needed. Free some space and run this again."
  }
}
Check "Disk space" ($free -lt 0 -or $free -ge 6) $freeText

# ---------------------------------------------------------------- 2. Python
Step "Checking for Python"
$py = Find-Python
if (-not $py) {
  Warn "Python not found. Installing Python 3.12 for this user (no admin needed)..."
  $installed = $false
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
      winget install --id Python.Python.3.12 -e --scope user --silent --accept-source-agreements --accept-package-agreements | Out-Null
      if ($LASTEXITCODE -eq 0) { $installed = $true }
    } catch { Warn "winget could not install Python: $_" }
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
Check "Python 3.10 or newer" $true $py

# ---------------------------------------------------------------- 3. Ollama
$ollamaExe = $null
if ($Eng -eq "ollama") {
  Step "Checking for Ollama (the local AI engine)"
  $ollamaExe = Find-Ollama
  if (-not $ollamaExe) {
    Warn "Ollama not found. Installing..."
    $done = $false
    if (Get-Command winget -ErrorAction SilentlyContinue) {
      try {
        winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
        if ($LASTEXITCODE -eq 0) { $done = $true }
      } catch { Warn "winget could not install Ollama: $_" }
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
  $engineUp = $false
  try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; $engineUp = $true }
  catch {
    Warn "Starting the Ollama engine..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    $tries = 0
    while ($tries -lt 30 -and -not $engineUp) {
      Start-Sleep -Seconds 2
      try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; $engineUp = $true } catch { $tries++ }
    }
  }
  Check "Ollama engine running" $engineUp "http://localhost:11434"
}

# ---------------------------------------------------------------- 4. Gauge
Step "Installing Gauge into its own Python environment"
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Could not create the Python environment (.venv). See $Log." }
& $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
& $venvPy -m pip install -e ".[ml]" --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw "Installing Gauge failed (pip exit code $LASTEXITCODE). Check your internet connection and run this again. Details: $Log" }
& $venvPy -m gauge.cli --version | Out-Null
Check "Gauge installed" ($LASTEXITCODE -eq 0) ".venv"
Ok "Gauge installed"

# ---------------------------------------------------------------- 5. Models
if ($Eng -eq "ollama") {
  Step "Downloading the AI models (the slow part; a few GB)"
  $wanted = @($Model, $FastModel, $EmbedModel) | Select-Object -Unique
  # A model already on this computer is left alone: nothing is downloaded twice.
  $present = @()
  try { $present = @((Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5).models | ForEach-Object { [string]$_.name }) } catch {}
  foreach ($m in $wanted) {
    if (($present -contains $m) -or ($present -contains ($m + ":latest"))) {
      Ok "$m is already on this computer"
      Check "Model $m" $true "already present"
      continue
    }
    Write-Host "    pulling $m ..."
    & $ollamaExe pull $m
    $pulled = ($LASTEXITCODE -eq 0)
    if (-not $pulled) { Warn "could not pull $m; Gauge still installs. Later, run: ollama pull $m" }
    Check "Model $m" $pulled "ollama pull $m"
  }
}

# ---------------------------------------------------------------- 6. Settings and a real test
Step "Setting up Gauge"
# Settings are written by Gauge itself, never by PowerShell: Windows PowerShell saves UTF-8 files with a
# byte-order mark, and a hand-rewritten config.json is the easiest way to break an install.
if ($Keep) {
  Ok "Your existing settings were kept ($ConfigPath)"
  Check "Settings kept" $true $ConfigPath
} elseif ($Eng -eq "native") {
  & $venvPy -m gauge.cli init --provider native | Out-Null
  Check "Settings written" ($LASTEXITCODE -eq 0) $ConfigPath
} else {
  & $venvPy -m gauge.cli init --provider ollama --model $Model | Out-Null
  $initOk = ($LASTEXITCODE -eq 0)
  & $venvPy -m gauge.cli settings set llm.fast_model $FastModel | Out-Null
  Check "Settings written" ($initOk -and $LASTEXITCODE -eq 0) $ConfigPath
}
& $venvPy -m gauge.cli doctor

# The check that matters: does the AI actually answer? (The first answer loads the model, so it can be slow.)
if ($Eng -eq "ollama") {
  Step "Asking the AI a test question"
  $answered = $false
  try {
    $body = @{ model = $Model; prompt = "Reply with the single word: ready"; stream = $false; options = @{ num_predict = 8 } } | ConvertTo-Json -Depth 4
    $reply = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    if ($null -ne $reply -and $null -ne $reply.response) { $answered = $true }
  } catch { Warn "No answer: $($_.Exception.Message)" }
  if ($answered) { Ok "The AI answered." }
  Check "The AI answers ($Model)" $answered "open the Ollama app, then run: ollama run $Model"
}

# ---------------------------------------------------------------- 7. Shortcut, checklist, launch
if (-not $NoShortcut) {
  Step "Creating the Desktop shortcut"
  try {
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
    Check "Desktop shortcut" $true $lnk -Optional
  } catch {
    Warn "Could not create the shortcut: $_"
    Check "Desktop shortcut" $false "open 'Gauge Studio.bat' in $Root instead" -Optional
  }
}

Write-Host ""
Write-Host "What was installed" -ForegroundColor Cyan
$failed = 0
foreach ($c in $Checks) {
  if ($c.Passed) { Write-Host ("  [ OK ] " + $c.Name) -ForegroundColor Green }
  elseif ($c.Optional) { Write-Host ("  [ -- ] " + $c.Name + " (optional)   -> " + $c.Note) -ForegroundColor Yellow }
  else { $failed++; Write-Host ("  [FAIL] " + $c.Name + "   -> " + $c.Note) -ForegroundColor Yellow }
}
Write-Host ""
Write-Host "Gauge is in: $Root"
if ($Eng -eq "native") {
  Write-Host "You chose GaugeLM, the built-in model. It has to be trained before it can write:" -ForegroundColor Yellow
  Write-Host "  .\gauge.bat corpus fetch --source gutenberg --query adventure"
  Write-Host "  .\gauge.bat native train --size tiny --minutes 240"
}
try { Stop-Transcript | Out-Null } catch {}
if ($failed -gt 0) {
  Write-Host "$failed step(s) need attention (see above). Everything else is installed. Log: $Log" -ForegroundColor Yellow
  exit 1
}
Write-Host "All done. Double-click 'Gauge Studio' on your Desktop any time." -ForegroundColor Green
if (-not $NoLaunch -and $Eng -ne "native") { Start-Process -FilePath (Join-Path $Root "Gauge Studio.bat") -WorkingDirectory $Root }
exit 0
