@echo off
REM Keep Gauge learning in a console window. Close the window or press Ctrl+C to stop.
cd /d "%~dp0.."
call gauge.bat daemon %*
pause
