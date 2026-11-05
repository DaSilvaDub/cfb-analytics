param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$WindowDays = 4
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'publish_guard.ps1')
Assert-PublicationAllowed $RepoRoot
$startedUtc = [datetime]::UtcNow
Set-Location $RepoRoot

$tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $tz)
$todayStr = $nowEt.ToString("yyyy-MM-dd")
$stampHHMM = $nowEt.ToString("HHmm")

$reportDir = Join-Path $RepoRoot "data\reports\$todayStr"
$snapDir = Join-Path $reportDir "snapshots"
New-Item -ItemType Directory -Force -Path $snapDir | Out-Null
$logFile = Join-Path $reportDir "$todayStr-pack-refresh.log"

function Write-Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$ts | [$stampHHMM] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

# Runs one step and stops the whole refresh on a non-zero exit code, so a failed
# ingest (e.g. Outlier HTTP 403) never flows into the board, sims, or Drive sync.
Write-Log "=== Starting Intraday CFB Pack Refresh ($todayStr @ $stampHHMM ET) ==="

$pythonExe = "C:\Users\dasil\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "daily", "--with-props")

$boardPath = Join-Path $reportDir "$todayStr-board-reasoning.txt"
$mispricedPath = Join-Path $reportDir "$todayStr-mispriced.txt"
$overBoardPath = Join-Path $reportDir "$todayStr-over-board-preview.txt"

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "board", "--date", $todayStr, "--with-reasoning") -OutputPath $boardPath
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "mispriced", "--date", $todayStr, "--min-edge", "0.01") -OutputPath $mispricedPath
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "over-board", "--date", $todayStr) -OutputPath $overBoardPath

Invoke-CheckedNative -Executable $pythonExe -Arguments @("scripts\run_master_v2_pipeline.py", "--date", $todayStr, "--window-days", $WindowDays)

# Archive timestamped snapshots so intraday line/prop movement can be diffed against 04:00 AM.
# -ErrorAction Stop: a failed snapshot copy ends the script with exit 1 instead of being skipped.
$simJsonPath = Join-Path $reportDir "$todayStr-master-v2-sim.json"
$slateTxtPath = Join-Path $reportDir "$todayStr-slate.txt"

if (Test-Path $simJsonPath) {
    Copy-Item -Path $simJsonPath -Destination (Join-Path $snapDir "$stampHHMM-master-v2-sim.json") -Force -ErrorAction Stop
}
if (Test-Path $slateTxtPath) {
    Copy-Item -Path $slateTxtPath -Destination (Join-Path $snapDir "$stampHHMM-slate.txt") -Force -ErrorAction Stop
}
if (Test-Path $overBoardPath) {
    Copy-Item -Path $overBoardPath -Destination (Join-Path $snapDir "$stampHHMM-over-board-preview.txt") -Force -ErrorAction Stop
}

Invoke-CheckedNative -Executable "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts/sync_to_drive.ps1"), "-RepoRoot", $RepoRoot, "-SlateDate", $todayStr, "-RunStartedUtc", $startedUtc.ToString("o"))

Write-Log "=== Completed Intraday CFB Pack Refresh ($todayStr @ $stampHHMM ET) ==="
exit 0
