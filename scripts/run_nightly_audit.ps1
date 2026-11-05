param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$AuditDate = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'publish_guard.ps1')
Assert-PublicationAllowed $RepoRoot
$startedUtc = [datetime]::UtcNow
Set-Location $RepoRoot

$tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $tz)

if ([string]::IsNullOrWhiteSpace($AuditDate)) {
    if ($nowEt.Hour -lt 12) {
        $AuditDate = $nowEt.AddDays(-1).ToString("yyyy-MM-dd")
    } else {
        $AuditDate = $nowEt.ToString("yyyy-MM-dd")
    }
}

$reportDir = Join-Path $RepoRoot "data\reports\$AuditDate"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$logFile = Join-Path $reportDir "$AuditDate-nightly-audit.log"

function Write-Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$ts | $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== Starting Nightly Postgame Audit & Settlement for Slate $AuditDate ==="

$pythonExe = "C:\Users\dasil\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Log "Step 1: Ingesting final scores, team_game_box, and player_game_passing (daily --with-scoring)..."
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "daily", "--with-scoring")

Write-Log "Step 2: Settling over-board snapshots for $AuditDate (settle --date $AuditDate)..."
Invoke-CheckedNative -Executable $pythonExe -Arguments @("-m", "cfb_analytics.cli", "settle", "--date", $AuditDate)

Write-Log "Step 3: Running Master Pipeline V2.0 Nightly Postgame Audit Engine (run_nightly_audit.py --date $AuditDate)..."
Invoke-CheckedNative -Executable $pythonExe -Arguments @("scripts\run_nightly_audit.py", "--date", $AuditDate)

Write-Log "Step 4: Syncing audit digest to Google Drive..."
Invoke-CheckedNative -Executable "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts/sync_to_drive.ps1"), "-RepoRoot", $RepoRoot, "-SlateDate", $AuditDate, "-RunStartedUtc", $startedUtc.ToString("o"))

Write-Log "=== Completed Nightly Postgame Audit & Settlement for Slate $AuditDate ==="
