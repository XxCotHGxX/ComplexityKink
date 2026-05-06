param(
    [int]$MaxParallel = 12,
    [int]$NBoot = 500,
    [int]$NCIBoot = 1000,
    [int]$NPlacebo = 500,
    [string]$ScoredDir = "data/stage_d/scored_combined",
    [string]$Rubric = "data/stage_d/ensemble_scores_current_aggregated.jsonl",
    [string]$Prompts = "data/stage_d/stage_d_prompts.jsonl",
    [string]$OutRoot = "results/stage_d_full_bootstrap_parallel",
    [string]$LogDir = "data/stage_d/logs",
    [int]$StartDelaySeconds = 6,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-SafeName {
    param([string]$Name)
    return ($Name -replace '[^A-Za-z0-9._-]', '_')
}

function Start-ModelJob {
    param([string]$Model)

    $safe = Get-SafeName $Model
    $outdir = Join-Path $OutRoot $safe
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null

    $stdout = Join-Path $LogDir "stage_d_parallel_$safe.stdout.log"
    $stderr = Join-Path $LogDir "stage_d_parallel_$safe.stderr.log"
    if ($Force) {
        Remove-Item -LiteralPath $stdout -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stderr -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath (Join-Path $outdir "analysis_summary.json") -ErrorAction SilentlyContinue
    }

    $proc = Start-Process -FilePath python -ArgumentList @(
        "-u",
        "src/analyze_kink.py",
        "--scored-dir", $ScoredDir,
        "--rubric", $Rubric,
        "--prompts", $Prompts,
        "--outdir", $outdir,
        "--min-rows", "5000",
        "--include-model", $Model,
        "--skip-combined",
        "--skip-visualizations",
        "--n-boot", "$NBoot",
        "--n-ci-boot", "$NCIBoot",
        "--n-placebo", "$NPlacebo"
    ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

    return [pscustomobject]@{
        ProcessId = $proc.Id
        Model = $Model
        SafeName = $safe
        Outdir = $outdir
        Stdout = $stdout
        Stderr = $stderr
        StartedAt = Get-Date
    }
}

$models = Get-ChildItem $ScoredDir -Filter *.jsonl |
    Sort-Object Name |
    ForEach-Object { $_.BaseName }

$pending = New-Object System.Collections.Queue
foreach ($model in $models) {
    $safe = Get-SafeName $model
    $summary = Join-Path (Join-Path $OutRoot $safe) "analysis_summary.json"
    if ((Test-Path $summary) -and -not $Force) {
        Write-Host "Skipping completed model: $model"
        continue
    }
    $pending.Enqueue($model)
}

$running = @()
$completed = @()
$failed = @()
$manifestPath = Join-Path $LogDir "stage_d_parallel_bootstrap_manifest.csv"
$statusPath = Join-Path $LogDir "stage_d_parallel_bootstrap_status.csv"

Write-Host "Starting Stage D per-model bootstrap pool: $($pending.Count) pending, max parallel $MaxParallel"

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $model = [string]$pending.Dequeue()
        $job = Start-ModelJob $model
        $running += $job
        Write-Host "Started $($job.Model) as PID $($job.ProcessId)"
        if ($StartDelaySeconds -gt 0) {
            Start-Sleep -Seconds $StartDelaySeconds
        }
    }

    $running | Export-Csv $manifestPath -NoTypeInformation
    $statusRows = @()
    foreach ($job in $running) {
        $proc = Get-Process -Id $job.ProcessId -ErrorAction SilentlyContinue
        $summary = Join-Path $job.Outdir "analysis_summary.json"
        $statusRows += [pscustomobject]@{
            ProcessId = $job.ProcessId
            Model = $job.Model
            Running = [bool]$proc
            CPU = if ($proc) { [math]::Round($proc.CPU, 1) } else { $null }
            WorkingSetMB = if ($proc) { [math]::Round($proc.WorkingSet64 / 1MB, 0) } else { $null }
            SummaryExists = Test-Path $summary
            Stdout = $job.Stdout
            Stderr = $job.Stderr
        }
    }
    $statusRows | Export-Csv $statusPath -NoTypeInformation

    Start-Sleep -Seconds 15

    $stillRunning = @()
    foreach ($job in $running) {
        $proc = Get-Process -Id $job.ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            $stillRunning += $job
            continue
        }

        $summary = Join-Path $job.Outdir "analysis_summary.json"
        if (Test-Path $summary) {
            $completed += $job
            Write-Host "Completed $($job.Model)"
        } else {
            $failed += $job
            Write-Host "Failed $($job.Model); see $($job.Stderr)"
        }
    }
    $running = $stillRunning
}

Write-Host "Pool finished: $($completed.Count) completed, $($failed.Count) failed"
if ($failed.Count -gt 0) {
    $failed | Format-Table -AutoSize
    exit 1
}
