param(
    [int]$Port = 7870,
    [int]$LinkPort = 7871,
    [int]$McpPort = 7872,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path.TrimEnd("\")
$Channel = Join-Path $Root "cli_brain_channel"
$CurrentPid = $PID

function Say($Message) {
    Write-Host "  - $Message"
}

function Is-UnderRoot($Path) {
    try {
        $Resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
        return $Resolved.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
    }
}

function Stop-DeckProcess($TargetPid, $Reason) {
    if (-not $TargetPid -or [int]$TargetPid -eq $CurrentPid) { return }
    try {
        $Proc = Get-Process -Id $TargetPid -ErrorAction Stop
        if ($DryRun) {
            Say "would stop PID $TargetPid ($($Proc.ProcessName)) - $Reason"
        } else {
            Say "stopping PID $TargetPid ($($Proc.ProcessName)) - $Reason"
            Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
        }
    } catch {
        # Process may already be gone.
    }
}

function Add-Descendants($ProcessList, $SeedPids) {
    $Wanted = New-Object "System.Collections.Generic.HashSet[int]"
    $Queue = New-Object "System.Collections.Generic.Queue[int]"
    foreach ($Seed in $SeedPids) {
        if ($Seed -and $Wanted.Add([int]$Seed)) {
            $Queue.Enqueue([int]$Seed)
        }
    }
    while ($Queue.Count -gt 0) {
        $Parent = $Queue.Dequeue()
        foreach ($Child in $ProcessList | Where-Object { $_.ParentProcessId -eq $Parent }) {
            if ($Wanted.Add([int]$Child.ProcessId)) {
                $Queue.Enqueue([int]$Child.ProcessId)
            }
        }
    }
    return @($Wanted)
}

function Get-Ancestors($ProcessList, $StartPid) {
    $Protected = New-Object "System.Collections.Generic.HashSet[int]"
    $Cursor = [int]$StartPid
    while ($Cursor -and $Protected.Add($Cursor)) {
        $Parent = $ProcessList | Where-Object { $_.ProcessId -eq $Cursor } | Select-Object -First 1
        if (-not $Parent -or -not $Parent.ParentProcessId) { break }
        $Cursor = [int]$Parent.ParentProcessId
    }
    return $Protected
}

function Expand-CommandLineText($CommandLine) {
    $Text = [string]$CommandLine
    if (-not $Text) { return "" }
    $Expanded = $Text
    $Matches = [regex]::Matches($Text, "(?i)-(?:e|ec|enc|encodedcommand)\s+([A-Za-z0-9+/=]+)")
    foreach ($Match in $Matches) {
        try {
            $Decoded = [System.Text.Encoding]::Unicode.GetString(
                [System.Convert]::FromBase64String($Match.Groups[1].Value))
            if ($Decoded) {
                $Expanded += "`n" + $Decoded
            }
        } catch {
            # Ignore malformed or truncated encoded commands.
        }
    }
    return $Expanded
}

Say "deck root: $Root"

$SeedPids = New-Object "System.Collections.Generic.HashSet[int]"

try {
    $PortPids = Get-NetTCPConnection -LocalPort @($Port, $LinkPort, $McpPort) -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($PortPid in $PortPids) {
        [void]$SeedPids.Add([int]$PortPid)
    }
} catch {
    Say "could not inspect port listeners: $($_.Exception.Message)"
}

$Processes = @()
try {
    $Processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
} catch {
    Say "could not inspect command lines; port cleanup still ran: $($_.Exception.Message)"
}

$Connected = Join-Path $Channel "connected"
if ((Test-Path -LiteralPath $Connected) -and (Is-UnderRoot $Connected)) {
    Get-ChildItem -LiteralPath $Connected -Filter "*.json" -File -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $Presence = Get-Content -LiteralPath $_.FullName -Raw -ErrorAction Stop | ConvertFrom-Json
            if ($Presence.pid) {
                [void]$SeedPids.Add([int64]$Presence.pid)
            }
        } catch {
            # Legacy heartbeat files may not carry pid metadata.
        }
    }
}

if ($Processes.Count -gt 0) {
    $ProtectedPids = Get-Ancestors -ProcessList $Processes -StartPid $CurrentPid
    $RootLower = $Root.ToLowerInvariant()
    $RootSlash = $RootLower.Replace("\", "/")
    $ChannelLower = $Channel.ToLowerInvariant()
    $CandidateNames = @(
        "python.exe",
        "pythonw.exe",
        "node.exe",
        "claude.exe",
        "gemini.exe",
        "codex.exe",
        "powershell.exe",
        "pwsh.exe"
    )
    $UniqueFiles = @("forum_daemon.py", "forum_codex_agent.py", "continuum_link_server.py", "continuum_mcp_server.py")
    $PresenceFiles = @("codex.json", "claude.json", "gemini.json")

    foreach ($Proc in $Processes) {
        if ($ProtectedPids.Contains([int]$Proc.ProcessId)) { continue }
        $Name = ([string]$Proc.Name).ToLowerInvariant()
        if ($CandidateNames -notcontains $Name) { continue }
        $Cmd = (Expand-CommandLineText $Proc.CommandLine).ToLowerInvariant()
        if (-not $Cmd) { continue }

        $DeckScoped = $Cmd.Contains($RootLower) -or $Cmd.Contains($RootSlash) -or $Cmd.Contains($ChannelLower)
        foreach ($File in $UniqueFiles) {
            if ($Cmd.Contains($File)) { $DeckScoped = $true }
        }
        foreach ($File in $PresenceFiles) {
            if ($Cmd.Contains($File) -and $Cmd.Contains("cli_brain_channel")) { $DeckScoped = $true }
        }
        if ($DeckScoped) {
            [void]$SeedPids.Add([int]$Proc.ProcessId)
        }
    }
}

if ($SeedPids.Count -gt 0) {
    $SeedArray = @()
    foreach ($SeedPid in $SeedPids) {
        $SeedArray += [int]$SeedPid
    }
    $AllPids = Add-Descendants -ProcessList $Processes -SeedPids $SeedArray
    foreach ($ProcessIdToStop in ($AllPids | Sort-Object -Descending)) {
        Stop-DeckProcess -TargetPid $ProcessIdToStop -Reason "Champion Continuum clean slate"
    }
} else {
    Say "no existing deck/daemon processes found"
}

if (Test-Path -LiteralPath $Channel) {
    if (-not (Is-UnderRoot $Channel)) {
        Say "refusing to clean channel outside deck root: $Channel"
        exit 1
    }

    $Patterns = @(
        "PENDING.json",
        "req_*.json",
        "resp_*.txt",
        "claim_*",
        "connected.json",
        "forum_state.json"
    )
    foreach ($Pattern in $Patterns) {
        Get-ChildItem -LiteralPath $Channel -Filter $Pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            if ($DryRun) {
                Say "would remove $($_.FullName)"
            } else {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }

    if ((Test-Path -LiteralPath $Connected) -and (Is-UnderRoot $Connected)) {
        Get-ChildItem -LiteralPath $Connected -Filter "*.json" -File -ErrorAction SilentlyContinue | ForEach-Object {
            if ($DryRun) {
                Say "would remove $($_.FullName)"
            } else {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }

    $Runs = Join-Path $Channel "runs"
    if ((Test-Path -LiteralPath $Runs) -and (Is-UnderRoot $Runs)) {
        Get-ChildItem -LiteralPath $Runs -Force -ErrorAction SilentlyContinue | ForEach-Object {
            if ($DryRun) {
                Say "would remove $($_.FullName)"
            } else {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Say "cleared volatile channel state; shared_store was left intact"
} else {
    Say "channel not found yet; nothing to clear"
}
