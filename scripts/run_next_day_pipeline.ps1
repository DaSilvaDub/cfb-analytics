param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$DaysAhead = 1,
    [int]$WindowDays = 4,
    [switch]$ForceIngest
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
$targetStr = $nowEt.AddDays($DaysAhead).ToString("yyyy-MM-dd")
$stampHHMM = $nowEt.ToString("HHmm")

$reportDir = Join-Path $RepoRoot "data\reports\$targetStr"
$snapDir = Join-Path $reportDir "snapshots"
New-Item -ItemType Directory -Force -Path $snapDir | Out-Null
$logFile = Join-Path $reportDir "$targetStr-early-bird-run.log"

function Write-Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$ts | [T-1 $stampHHMM] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== Starting Day-Before Early-Bird Master Pipeline V2.0 Run for Target Slate: $targetStr ==="

$pythonExe = "C:\Users\dasil\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$dbPath = Join-Path $RepoRoot "data\cfb.sqlite3"
Invoke-CheckedNative $pythonExe @('-m', 'cfb_analytics.cli', 'doctor')
Invoke-CheckedNative $pythonExe @('-m', 'cfb_analytics.cli', 'schedule')
Invoke-CheckedNative $pythonExe @('-m', 'cfb_analytics.cli', 'daily', '--with-props', '--with-scoring')

Write-Log "Step 2: Generating Early-Bird board, mispriced, and over-board for $targetStr..."
$boardPath = Join-Path $reportDir "$targetStr-board-reasoning.txt"
$mispricedPath = Join-Path $reportDir "$targetStr-mispriced.txt"
$overBoardPath = Join-Path $reportDir "$targetStr-over-board-preview.txt"

Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "board", "--date", $targetStr, "--with-reasoning") -OutputPath $boardPath
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "mispriced", "--date", $targetStr, "--min-edge", "0.01") -OutputPath $mispricedPath
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "over-board", "--date", $targetStr) -OutputPath $overBoardPath

Write-Log "Step 3: Running 50,000-trial Master Pipeline V2.0 Monte Carlo simulation for $targetStr (window: $WindowDays days)..."
Invoke-CheckedNative -Executable $pythonExe -Arguments @("scripts\run_master_v2_pipeline.py", "--date", $targetStr, "--window-days", $WindowDays)

Assert-CfbSimulation (Join-Path $reportDir "$targetStr-master-v2-sim.json") $targetStr $startedUtc

# Archive immutable T-1 1800 baseline snapshots for overnight steam comparison at 4:00 AM
$simJsonPath = Join-Path $reportDir "$targetStr-master-v2-sim.json"
$slateTxtPath = Join-Path $reportDir "$targetStr-slate.txt"

$snapshots = @{
    $simJsonPath = 'Tminus1-1800-master-v2-sim.json'
    $slateTxtPath = 'Tminus1-1800-slate.txt'
    $overBoardPath = 'Tminus1-1800-over-board-preview.txt'
    $boardPath = 'Tminus1-1800-board-reasoning.txt'
    $mispricedPath = 'Tminus1-1800-mispriced.txt'
}
foreach ($source in $snapshots.Keys) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing snapshot source: $source" }
    $item = Get-Item -LiteralPath $source
    if ($item.Length -eq 0 -or $item.LastWriteTimeUtc -lt $startedUtc) { throw "Stale/empty snapshot source: $source" }
    if (Test-Path -LiteralPath (Join-Path $snapDir $snapshots[$source])) { throw 'Immutable T-1 snapshot already exists; refusing overwrite' }
}
foreach ($source in $snapshots.Keys) {
    Copy-Item -LiteralPath $source -Destination (Join-Path $snapDir $snapshots[$source]) -ErrorAction Stop
}

Write-Log "Step 4: Syncing Early-Bird reports and packs to Google Drive..."
Invoke-CheckedNative -Executable "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts/sync_to_drive.ps1"), "-RepoRoot", $RepoRoot, "-SlateDate", $targetStr, "-RunStartedUtc", $startedUtc.ToString("o"))

Write-Log "=== Completed Day-Before Early-Bird Master Pipeline V2.0 Run for $targetStr ==="
