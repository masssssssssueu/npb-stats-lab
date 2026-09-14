# タスクスケジューラから呼び出される、日次更新のラッパースクリプト。
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\daily_update.py"
