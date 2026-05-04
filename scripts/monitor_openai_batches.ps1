param(
    [string[]]$Models = @("gpt-4.1", "gpt-5-mini"),
    [int]$IntervalSeconds = 300,
    [string]$LogPath = "data/stage_d/logs/monitor_openai_batches.log"
)

$ErrorActionPreference = "Continue"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$LogFullPath = Join-Path $RepoRoot $LogPath
$LogDir = Split-Path -Parent $LogFullPath
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -LiteralPath $LogFullPath -Value $line -Encoding UTF8
}

function Get-SafeName {
    param([string]$Model)
    return ($Model -replace "[/:]", "_")
}

Write-Log "Starting OpenAI batch monitor for models: $($Models -join ', ')"

$done = @{}
foreach ($model in $Models) {
    $safe = Get-SafeName $model
    $sentinel = Join-Path $RepoRoot "data/stage_d/batch_state/$safe.retrieved"
    if (Test-Path -LiteralPath $sentinel) {
        $done[$model] = $true
        Write-Log "$model already has retrieval sentinel: $sentinel"
    } else {
        $done[$model] = $false
    }
}

while (($done.Values | Where-Object { -not $_ }).Count -gt 0) {
    foreach ($model in $Models) {
        if ($done[$model]) {
            continue
        }

        Write-Log "Polling $model"
        $statusOutput = & python scripts/openai_batch.py --state-dir data/stage_d/batch_state status --model $model 2>&1
        $statusText = ($statusOutput | Out-String).Trim()
        if ($statusText) {
            Write-Log $statusText
        }

        if ($statusText -match "Status:\s+completed") {
            Write-Log "$model completed; retrieving output"
            $retrieveOutput = & python scripts/openai_batch.py --gen-dir data/stage_d/generations --state-dir data/stage_d/batch_state --result-dir data/stage_d/batch_results retrieve --model $model 2>&1
            $retrieveCode = $LASTEXITCODE
            $retrieveText = ($retrieveOutput | Out-String).Trim()
            if ($retrieveText) {
                Write-Log $retrieveText
            }
            if ($retrieveCode -eq 0) {
                $safe = Get-SafeName $model
                $sentinel = Join-Path $RepoRoot "data/stage_d/batch_state/$safe.retrieved"
                Set-Content -LiteralPath $sentinel -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz") -Encoding UTF8
                $done[$model] = $true
                Write-Log "$model retrieved; sentinel written to $sentinel"
            } else {
                Write-Log "$model retrieve failed with exit code $retrieveCode; will retry on next poll"
            }
        } elseif ($statusText -match "Status:\s+(failed|cancelled|expired)") {
            $safe = Get-SafeName $model
            $failure = Join-Path $RepoRoot "data/stage_d/batch_state/$safe.failed"
            Set-Content -LiteralPath $failure -Value $statusText -Encoding UTF8
            $done[$model] = $true
            Write-Log "$model reached terminal failure state; marker written to $failure"
        }
    }

    if (($done.Values | Where-Object { -not $_ }).Count -gt 0) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}

Write-Log "OpenAI batch monitor finished."
