param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)][string]$SlateDate,
    [Parameter(Mandatory = $true)][datetime]$RunStartedUtc,
    [string]$DriveRoot = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'publish_guard.ps1')
Assert-PublicationAllowed $RepoRoot
Set-Location -LiteralPath $RepoRoot
if ((Get-PSDrive C).Free -le 20GB) { throw 'C: must have more than 20 GB free' }
if (Test-Path -LiteralPath (Join-Path $RepoRoot 'data/cfb.sqlite3-journal')) { throw 'Database transaction lock present' }
if ($RunStartedUtc.ToUniversalTime() -gt [datetime]::UtcNow -or $RunStartedUtc.ToUniversalTime() -lt [datetime]::UtcNow.AddHours(-2)) { throw 'Invalid or expired run start' }
$reportDir = Join-Path $RepoRoot "data/reports/$SlateDate"
Assert-CfbSimulation (Join-Path $reportDir "$SlateDate-master-v2-sim.json") $SlateDate $RunStartedUtc
if (-not $DriveRoot) {
    foreach ($candidate in @('C:/Users/dasil/My Drive (dasilvadub@gmail.com)', 'G:/My Drive')) {
        if (Test-Path -LiteralPath $candidate -PathType Container) { $DriveRoot = $candidate; break }
    }
}
if (-not $DriveRoot -or -not (Test-Path -LiteralPath $DriveRoot -PathType Container)) { throw 'Google Drive root missing' }
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$exportStarted = [datetime]::UtcNow
Invoke-CheckedNative $pythonExe @((Join-Path $RepoRoot 'scripts/export_ncaafb_extra_pack.py'), '--date', $SlateDate)
$csv = Join-Path $RepoRoot 'artifacts/exports/ncaafb_only.csv'
Assert-CfbCsv $csv $SlateDate $exportStarted
$reports = @(Get-ChildItem -LiteralPath $reportDir -File | Where-Object {
    $_.Name -match '^(\d{4}-\d{2}-\d{2})-(master-v2-digest\.md|over-board.*\.txt|board-reasoning\.txt|mispriced\.txt|slate\.txt)$'
})
if ($reports.Count -eq 0) { throw '0 report matches' }
$hashes = @{}
foreach ($file in @((Get-Item -LiteralPath $csv)) + $reports) {
    if ($file.LastWriteTimeUtc -lt $RunStartedUtc.ToUniversalTime() -or $file.Length -eq 0) { throw "Stale or empty artifact: $($file.FullName)" }
    if ($file.FullName -ne $csv -and -not $file.Name.StartsWith("$SlateDate-")) { throw 'Report slate-date mismatch' }
    $hashes[$file.FullName] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}
$stage = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    foreach ($path in $hashes.Keys) {
        $dest = Join-Path $stage (Split-Path -Leaf $path)
        Copy-Item -LiteralPath $path -Destination $dest -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -ne $hashes[$path]) { throw 'Artifact changed during staging' }
    }
    $today = Join-Path $DriveRoot 'today'
    $ncaa = Join-Path $DriveRoot 'BETTING REPORTS/NCAA'
    $dated = Join-Path $today "extra_packs_${SlateDate}_latest"
    $rolling = Join-Path $today 'extra_packs'
    New-Item -ItemType Directory -Force -Path $today, $ncaa, $dated, $rolling | Out-Null
    Copy-Item -LiteralPath (Join-Path $stage 'ncaafb_only.csv') -Destination $dated -Force
    Copy-Item -LiteralPath (Join-Path $stage 'ncaafb_only.csv') -Destination $rolling -Force
    foreach ($report in $reports) {
        $source = Join-Path $stage $report.Name
        Copy-Item -LiteralPath $source -Destination $today -Force
        Copy-Item -LiteralPath $source -Destination $ncaa -Force
    }
}
finally {
    Get-ChildItem -LiteralPath $stage -File | Remove-Item -Force
    Remove-Item -LiteralPath $stage -Force
}
Write-Output "UNPROMOTED - shadow output, not decision-grade. NCAA sync completed for $SlateDate."
