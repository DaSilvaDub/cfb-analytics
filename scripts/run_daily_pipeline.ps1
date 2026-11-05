param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$WindowDays = 4
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "publish_guard.ps1")
Assert-PublicationAllowed $RepoRoot
$startedUtc = [datetime]::UtcNow
if ((Get-PSDrive C).Free -le 20GB) { throw "C: must have more than 20 GB free" }
if (Test-Path -LiteralPath (Join-Path $RepoRoot 'data/cfb.sqlite3-journal')) { throw 'Database transaction lock present' }
Set-Location $RepoRoot

$tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $tz)
$todayStr = $nowEt.ToString("yyyy-MM-dd")
$yesterdayStr = $nowEt.AddDays(-1).ToString("yyyy-MM-dd")

$reportDir = Join-Path $RepoRoot "data\reports\$todayStr"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$logFile = Join-Path $reportDir "$todayStr-scheduled-run.log"

function Write-Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$ts | $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

# Runs one step and stops the whole run on a non-zero exit code, so a failed
# fetch (e.g. Outlier HTTP 403) never flows into the board, sims, or Drive sync.
Write-Log "=== Starting NCAA Football Master Pipeline V2.0 Daily Run ($todayStr) ==="

$pythonExe = "C:\Users\dasil\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "doctor")
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "schedule")

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "daily", "--with-props", "--with-scoring")

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "settle", "--date", $yesterdayStr)

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "board", "--date", $todayStr, "--with-reasoning") -OutputPath (Join-Path $reportDir "$todayStr-board-reasoning.txt")
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "mispriced", "--date", $todayStr, "--min-edge", "0.01") -OutputPath (Join-Path $reportDir "$todayStr-mispriced.txt")
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "over-board", "--date", $todayStr) -OutputPath (Join-Path $reportDir "$todayStr-over-board-preview.txt")

Invoke-CheckedNative -Executable $pythonExe -Arguments @("scripts\run_master_v2_pipeline.py", "--date", $todayStr, "--window-days", $WindowDays)

Assert-CfbSimulation (Join-Path $reportDir "$todayStr-master-v2-sim.json") $todayStr $startedUtc

Invoke-CheckedNative -Executable "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts/sync_to_drive.ps1"), "-RepoRoot", $RepoRoot, "-SlateDate", $todayStr, "-RunStartedUtc", $startedUtc.ToString("o"))

Write-Log "=== Completed NCAA Football Master Pipeline V2.0 Daily Run ($todayStr) ==="
exit 0
