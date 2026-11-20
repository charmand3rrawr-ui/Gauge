<#
.SYNOPSIS
  Register Gauge's learning cycle as a Windows Scheduled Task (runs every N hours, even after reboot).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\install-task.ps1 -Hours 6
  powershell -ExecutionPolicy Bypass -File .\scripts\install-task.ps1 -Hours 6 -Project "My Serial" -Write
  powershell -ExecutionPolicy Bypass -File .\scripts\install-task.ps1 -Remove
#>
param(
  [int]$Hours = 6,
  [string]$Project = "",
  [switch]$Write,
  [switch]$Remove
)
$name = "Gauge Learning Cycle"
if ($Remove) { Unregister-ScheduledTask -TaskName $name -Confirm:$false; Write-Host "Removed task '$name'"; exit 0 }

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$bat = Join-Path $root "gauge.bat"
$args = "daemon --once"
if ($Project) { $args += " --project `"$Project`"" }
if ($Write) { $args += " --write" }

$action = New-ScheduledTaskAction -Execute $bat -Argument $args -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Hours $Hours)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours ([math]::Max(1, $Hours - 1)))
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Registered '$name': runs '$bat $args' every $Hours hour(s). Manage it in Task Scheduler."
