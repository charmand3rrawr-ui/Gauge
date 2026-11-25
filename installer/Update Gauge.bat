@echo off
setlocal
title Update Gauge
REM ---------------------------------------------------------------------------------------------
REM  Update Gauge: brings an installed Gauge up to date. Double-click it.
REM
REM  It only fetches Gauge's own program (about 1 MB), and only when there is a newer version.
REM  It does NOT reinstall Python or Ollama and does NOT download the AI models again.
REM  Your stories, your library and your settings are left exactly as they are.
REM
REM  It finds Gauge in  %USERPROFILE%\Gauge , or next to this file. Installed somewhere else?
REM     set GAUGE_DIR=D:\Gauge     before running it.
REM  Not installed yet? Use "Get Gauge.bat" instead.
REM ---------------------------------------------------------------------------------------------
set "GAUGE_SELF=%~f0"
if not defined GAUGE_BRANCH set "GAUGE_BRANCH=ccr-24dc418f-hsneyx"
echo.
echo   Checking for a newer version of Gauge...
echo.
REM Everything from here on sits in one block. cmd reads a whole block before running any of it, so
REM this file can safely replace itself with the new version while it is running.
(
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:GAUGE_SELF); Invoke-Expression $s.Substring($s.LastIndexOf('#---POWERSHELL---'))"
  if errorlevel 1 (
    echo.
    echo   The update did not finish. Nothing is broken: Gauge still works as it did before.
    echo   You can run this file again. If it keeps failing, take a photo of this window and ask for help.
    echo.
    pause
    exit /b 1
  )
  echo.
  pause
  exit /b 0
)
#---POWERSHELL---
$ErrorActionPreference = "Stop"
try {
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
  $ProgressPreference = "SilentlyContinue"
  $repo = "charmand3rrawr-ui/Gauge"

  # ---- 1. Where is Gauge installed?
  $here = Split-Path -Parent $env:GAUGE_SELF
  $places = @($env:GAUGE_DIR, $here, (Split-Path -Parent $here), (Join-Path $env:USERPROFILE "Gauge")) | Where-Object { $_ }
  $dir = $null
  foreach ($p in $places) {
    if ((Test-Path (Join-Path $p "pyproject.toml")) -and (Test-Path (Join-Path $p "gauge"))) { $dir = $p; break }
  }
  if (-not $dir) { throw "Gauge is not installed on this computer yet. Run 'Get Gauge.bat' first; this file only updates." }
  Write-Host "  Gauge is in $dir"
  if (Test-Path (Join-Path $dir ".git")) { throw "The copy of Gauge in $dir is a git checkout. Update it with 'git pull'; this updater would overwrite your changes." }

  # ---- 2. Is there anything newer? One small question to GitHub; no download if the answer is no.
  $marker = Join-Path $dir "installer\installed-version.txt"
  $have = ""
  if (Test-Path $marker) { $have = ([IO.File]::ReadAllText($marker)).Trim() }
  $branches = @($env:GAUGE_BRANCH, "main") | Where-Object { $_ } | Select-Object -Unique
  $tmp = Join-Path $env:TEMP ("gauge-update-" + [Guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $tmp | Out-Null
  $src = $null; $branch = $null; $sha = ""; $n = 0
  foreach ($b in $branches) {
    $n++
    $latest = ""
    try { $latest = [string](Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits/$b" -TimeoutSec 20).sha } catch { $latest = "" }
    if ($latest -and ($have -eq "$b $latest")) {
      Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
      Write-Host ""
      Write-Host ("  Gauge is already up to date (version " + $latest.Substring(0, 7) + "). Nothing was downloaded.") -ForegroundColor Green
      exit 0
    }
    $zip = Join-Path $tmp "gauge-$n.zip"
    $out = Join-Path $tmp "unzipped-$n"
    Write-Host "  Downloading the new version of Gauge (about 1 MB) ..."
    try { Invoke-WebRequest -Uri "https://github.com/$repo/archive/refs/heads/$b.zip" -OutFile $zip -UseBasicParsing }
    catch { Write-Host ("    could not download it: " + $_.Exception.Message); continue }
    Expand-Archive -LiteralPath $zip -DestinationPath $out -Force
    $top = Get-ChildItem -LiteralPath $out -Directory | Select-Object -First 1
    if ($top -and (Test-Path (Join-Path $top.FullName "pyproject.toml")) -and (Test-Path (Join-Path $top.FullName "gauge"))) {
      $src = $top.FullName; $branch = $b; $sha = $latest; break
    }
    Write-Host "    that copy has no program in it; trying the next one"
  }
  if (-not $src) { throw "Could not download the update. Check your internet connection and run this file again." }

  # ---- 3. Put the new program in place. Data, stories, settings and the Python environment live in
  #         folders the download does not contain, so they are not touched.
  $oldDeps = ""
  if (Test-Path (Join-Path $dir "pyproject.toml")) { $oldDeps = (Get-FileHash -LiteralPath (Join-Path $dir "pyproject.toml") -Algorithm SHA256).Hash }
  Copy-Item -Path (Join-Path $src "*") -Destination $dir -Recurse -Force
  # Program files that no longer exist in the new version are removed, so an old module cannot linger.
  $newRoot = Join-Path $src "gauge"
  $oldRoot = Join-Path $dir "gauge"
  $removed = 0
  foreach ($f in (Get-ChildItem -LiteralPath $oldRoot -Recurse -Filter "*.py" -File)) {
    $rel = $f.FullName.Substring($oldRoot.Length).TrimStart("\")
    if (-not (Test-Path (Join-Path $newRoot $rel))) { Remove-Item -LiteralPath $f.FullName -Force; $removed++ }
  }
  $newDeps = (Get-FileHash -LiteralPath (Join-Path $dir "pyproject.toml") -Algorithm SHA256).Hash

  # ---- 4. Python packages: only when the list of packages Gauge needs has changed.
  $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    Write-Host "  The Python environment is missing. Run 'Install Gauge.bat' in the Gauge folder to rebuild it." -ForegroundColor Yellow
  } elseif ($oldDeps -ne $newDeps) {
    Write-Host "  This version needs different Python packages; installing them (a minute or two) ..."
    Push-Location $dir
    & $venvPy -m pip install -e ".[ml]" --quiet --disable-pip-version-check
    $pipCode = $LASTEXITCODE
    Pop-Location
    if ($pipCode -ne 0) { throw "The new program is in place, but its Python packages could not be installed. Check your internet connection and run this file again." }
  } else {
    Write-Host "  Python packages are unchanged; nothing to install."
  }

  # ---- 4b. An NVIDIA graphics card that Gauge's own model cannot use yet: set it up once. Skipped when it
  # is already set up, so nothing is downloaded twice.
  if ((Test-Path $venvPy) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    & $venvPy -c "import cupy" 2>$null
    if ($LASTEXITCODE -ne 0) {
      Write-Host "  Setting up the graphics card for Gauge's own model (a large download, once) ..."
      Push-Location $dir
      & $venvPy -m gauge.cli compute setup-gpu
      $gpuCode = $LASTEXITCODE
      Pop-Location
      if ($gpuCode -ne 0) { Write-Host "  The graphics card could not be set up; GaugeLM keeps using the processor. Try again from Gauge Studio: Switches." -ForegroundColor Yellow }
    }
  }

  # ---- 5. Remember which version this is, and say what changed.
  $markerDir = Split-Path -Parent $marker
  if (-not (Test-Path $markerDir)) { New-Item -ItemType Directory -Path $markerDir | Out-Null }
  if ($sha) { [IO.File]::WriteAllText($marker, "$branch $sha") }
  $oldSha = ""
  if ($have -and $have.Contains(" ")) { $oldSha = $have.Substring($have.IndexOf(" ") + 1) }
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host ""
  if ($sha) { Write-Host ("  Gauge is updated to version " + $sha.Substring(0, 7) + ".") -ForegroundColor Green }
  else { Write-Host "  Gauge is updated." -ForegroundColor Green }
  if ($removed -gt 0) { Write-Host "  Removed $removed old program file(s) that the new version no longer uses." }
  if ($oldSha -and $sha) {
    try {
      $cmp = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/compare/$oldSha...$sha" -TimeoutSec 20
      $lines = @($cmp.commits | ForEach-Object { ([string]$_.commit.message -split "`n")[0] })
      if ($lines.Count -gt 0) {
        Write-Host ""
        Write-Host "  What is new:"
        $lines | Select-Object -Last 8 | ForEach-Object { Write-Host ("    - " + $_) }
      }
    } catch {}
  }
  Write-Host ""
  Write-Host "  Not touched: Python, Ollama, the AI models, your stories, your library and your settings."
  Write-Host "  If Gauge Studio is open, close it and open it again to use the new version."
  exit 0
} catch {
  Write-Host ""
  Write-Host ("  ERROR: " + $_.Exception.Message) -ForegroundColor Red
  exit 1
}
