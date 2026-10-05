# Watch only: this script never starts or cancels a render.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Path,
    [ValidateRange(1, 2147483647)]
    [int]$PollMilliseconds = 500,
    [ValidateRange(1, 2147483647)]
    [int]$WaitSeconds = 120
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'
$activity = 'Ego video'
$snapshot = $null
$unreadableSince = [DateTimeOffset]::UtcNow
$readError = 'File has not appeared yet.'

try {
    while ($true) {
        try {
            # The writer replaces the file atomically. Retry sharing/read errors.
            $candidate = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($null -eq $candidate -or $candidate.status -notin @('running', 'completed', 'failed')) {
                throw 'No valid progress status in snapshot.'
            }
            $updatedAt = [DateTimeOffset]::Parse($candidate.updated_at)
            $snapshot = $candidate
            $unreadableSince = $null
        }
        catch {
            $readError = $_.Exception.Message
            if ($null -eq $unreadableSince) {
                $unreadableSince = [DateTimeOffset]::UtcNow
            }
        }

        $now = [DateTimeOffset]::UtcNow
        if ($null -ne $unreadableSince -and ($now - $unreadableSince).TotalSeconds -ge $WaitSeconds) {
            throw "Cannot read progress file '$Path' after $WaitSeconds seconds: $readError"
        }
        if ($null -eq $snapshot) {
            $waited = ($now - $unreadableSince).TotalSeconds
            Write-Progress -Id 1 -Activity $activity -Status 'Waiting for progress file' `
                -CurrentOperation ("{0} | waited {1:N1}s / {2}s" -f $Path, $waited, $WaitSeconds) `
                -PercentComplete -1
        }
        else {
            if ($snapshot.status -eq 'completed') {
                Write-Progress -Id 1 -Activity $activity -Completed
                Write-Host ("Completed: {0} (elapsed {1:N1}s)" -f $snapshot.message, $snapshot.elapsed_seconds)
                exit 0
            }
            if ($snapshot.status -eq 'failed') {
                throw ("Ego video failed: {0}" -f $snapshot.message)
            }

            $percent = -1
            $counts = ''
            if ($null -ne $snapshot.completed -and $null -ne $snapshot.total -and $snapshot.total -gt 0) {
                $percent = [int][Math]::Floor([Math]::Min(100, [Math]::Max(0, 100.0 * $snapshot.completed / $snapshot.total)))
                $counts = " | $($snapshot.completed)/$($snapshot.total)"
            }
            $age = [Math]::Max(0, ($now - $updatedAt).TotalSeconds)
            $details = if ($null -ne $snapshot.details) { $snapshot.details | ConvertTo-Json -Compress -Depth 10 } else { '{}' }
            $operation = "elapsed {0:N1}s | last update {1:N1}s ago | {2}" -f $snapshot.elapsed_seconds, $age, $details
            if ($null -ne $unreadableSince) {
                $operation += " | retrying read: $readError"
            }
            Write-Progress -Id 1 -Activity ("{0}: {1}" -f $activity, $snapshot.stage) `
                -Status ("{0}{1}" -f $snapshot.message, $counts) `
                -CurrentOperation $operation -PercentComplete $percent
        }
        Start-Sleep -Milliseconds $PollMilliseconds
    }
}
catch {
    Write-Progress -Id 1 -Activity $activity -Completed
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
