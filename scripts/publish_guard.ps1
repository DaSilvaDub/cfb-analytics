$ErrorActionPreference = 'Stop'

function Assert-PublicationAllowed([string]$Root) {
    $holds = @((Join-Path $Root 'docs/remediation/publication_hold.json'),
        'C:/Users/dasil/outlier-quarantine/2026-11-05/publication_hold.json')
    foreach ($path in $holds) {
        if (Test-Path -LiteralPath $path) {
            $hold = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
            if ($hold.active -isnot [bool]) { throw "Invalid publication hold: $path" }
            if ($hold.active) { throw "Publication withheld: $($hold.reason) [$path]" }
        }
    }
}

function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$Arguments, [string]$OutputPath = '')
    # Process redirection avoids PowerShell 5.1 treating native stderr logging as
    # a terminating NativeCommandError, while all PowerShell errors remain Stop.
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.WorkingDirectory = (Get-Location).ProviderPath
    $quoted = foreach ($arg in $Arguments) {
        '"' + ([regex]::Replace([regex]::Replace($arg, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1')) + '"'
    }
    $info.Arguments = $quoted -join ' '
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.CreateNoWindow = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw "Could not start $Executable" }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $out = $stdout.Result
        $err = $stderr.Result
        if ($process.ExitCode -ne 0) { throw "$Executable exited $($process.ExitCode): $err $out" }
        if (($err + "`n" + $out) -match '(?i)(HTTP\s*403|ENOSPC|Errno\s*28|database is locked|database.*malformed|slate.date.mismatch|\b0 (?:matches|matched games)\b)') {
            throw "Producer reported fatal health failure: $err $out"
        }
        if ($OutputPath) { $out | Set-Content -LiteralPath $OutputPath -Encoding UTF8 }
        else { Write-Output $out }
        if ($err) { [Console]::Error.WriteLine($err) }
    }
    finally { $process.Dispose() }
}

function Assert-CfbSimulation([string]$Path, [string]$Date, [datetime]$StartedUtc) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing simulation: $Path" }
    $sim = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ($sim.base_date -ne $Date) { throw 'Simulation slate-date mismatch' }
    if (-not $sim.generated_utc -or ([datetime]$sim.generated_utc).ToUniversalTime() -lt $StartedUtc.ToUniversalTime()) { throw 'Stale simulation' }
    if (@($sim.games).Count -eq 0) { throw '0 matched games in simulation' }
    $matched = @($sim.games | Where-Object { $_.football_date -eq $Date })
    if ($matched.Count -eq 0) { throw '0 matched games for requested slate date' }
    foreach ($game in $sim.games) { if (-not $game.football_date) { throw 'Simulation game lacks slate identity' } }
}

function Assert-CfbCsv([string]$Path, [string]$Date, [datetime]$StartedUtc) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing export: $Path" }
    if ((Get-Item -LiteralPath $Path).LastWriteTimeUtc -lt $StartedUtc.ToUniversalTime()) { throw 'Stale export' }
    $rows = @(Import-Csv -LiteralPath $Path)
    if ($rows.Count -eq 0) { throw '0 matches in export' }
    foreach ($row in $rows) {
        if (-not $row.kickoff_et -or ([datetimeoffset]$row.kickoff_et).ToString('yyyy-MM-dd') -ne $Date) { throw 'Export slate-date mismatch' }
        if (-not $row.game_id -or $row.game_id -notmatch '^cfbd:') { throw 'Export lacks canonical game identity' }
    }
}
